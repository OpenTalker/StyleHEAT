import os
import time
import glob
from tqdm import tqdm

from utils.distributed import master_only_print as print
from utils.distributed import master_only, get_rank
from utils.trainer import accumulate
from utils.meters import Meter, add_hparams
from utils.misc import to_cuda
from utils.lpips import LPIPS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BaseTrainer(object):
    r"""Base trainer. We expect that all trainers inherit this class.

    Args:
        opt (obj): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(
        self,
        opt,
        net_G,
        net_G_ema,
        opt_G,
        sch_G,
        train_data_loader,
        val_data_loader=None
    ):
        super(BaseTrainer, self).__init__()
        print('Setup trainer.')

        self.opt = opt
        self.net_G = net_G
        self.net_G_ema = net_G_ema
        self.opt_G = opt_G
        self.sch_G = sch_G

        if opt.distributed:
            self.net_G_module = self.net_G.module
        else:
            self.net_G_module = self.net_G

        self.is_inference = train_data_loader is None
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.criteria = nn.ModuleDict()
        self.weights = dict()
        self.losses = dict(gen_update=dict(), dis_update=dict())

        self.gen_losses = self.losses['gen_update']
        self._init_loss(opt)

        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and \
                self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')

        if self.is_inference:
            # The initialization steps below can be skipped during inference.
            return

        # Initialize logging attributes.
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.elapsed_iteration_time = 0
        self.time_iteration = -1
        self.time_epoch = -1
        if getattr(self.opt, 'speed_benchmark', False):
            self.accu_gen_forw_iter_time = 0
            self.accu_gen_loss_iter_time = 0
            self.accu_gen_back_iter_time = 0
            self.accu_gen_step_iter_time = 0
            self.accu_gen_avg_iter_time = 0

        # Initialize tensorboard and hparams.
        self._init_tensorboard()
        self._init_hparams()
        self.lpips = LPIPS()
        self.best_lpips = None

    def _init_tensorboard(self):
        r"""Initialize the tensorboard. Different algorithms might require
        different performance metrics. Hence, custom tensorboard
        initialization might be necessary.
        """
        # Logging frequency: self.opt.logging_iter
        self.meters = {}
        names = ['optim/gen_lr', 'time/iteration', 'time/epoch',
                 'metric/best_lpips', 'metric/lpips']
        for name in names:
            self.meters[name] = Meter(name)

        # Logging frequency: self.opt.image_display_iter
        self.image_meter = Meter('images')

    def _init_hparams(self):
        r"""Initialize a dictionary of hyperparameters that we want to monitor
        in the HParams dashboard in tensorBoard.
        """
        self.hparam_dict = {}

    def _write_tensorboard(self):
        r"""Write values to tensorboard. By default, we will log the time used
        per iteration, time used per epoch, generator learning rate, and
        discriminator learning rate. We will log all the losses as well as
        custom meters.
        """
        # Logs that are shared by all models.
        self._write_to_meters({'time/iteration': self.time_iteration,
                               'time/epoch': self.time_epoch,
                               'optim/gen_lr': self.sch_G.get_last_lr()[0]},
                              self.meters)
        # Logs for loss values. Different models have different losses.
        self._write_loss_meters()
        # Other custom logs.
        self._write_custom_meters()
        # Write all logs to tensorboard.
        self._flush_meters(self.meters)

    def _write_loss_meters(self):
        r"""Write all loss values to tensorboard."""
        for loss_name, loss in self.gen_losses.items():
            full_loss_name = 'gen_update' + '/' + loss_name
            if full_loss_name not in self.meters.keys():
                # Create a new meter if it doesn't exist.
                self.meters[full_loss_name] = Meter(full_loss_name)
            self.meters[full_loss_name].write(loss.item())

    def test_everything(self, train_dataset, val_dataset, current_epoch, current_iteration):
        r"""Test the functions defined in the models. by default, we will test the
        training function, the inference function, the visualization function.
        """
        self._set_custom_debug_parameter()
        self.start_of_epoch(current_epoch)
        print('Start testing your functions')
        for _ in tqdm(range(2)):
            data = iter(train_dataset).next()
            data = self.start_of_iteration(data, current_iteration)
            self.optimize_parameters(data, if_optimize=True)
            current_iteration += 1
            self.end_of_iteration(data, current_epoch, current_iteration)

        self.test(train_dataset, output_dir=os.path.join(self.opt.logdir, 'evaluation'),
                  current_iteration=current_iteration, test_limit=2)
        self.save_image(self._get_save_path('image', 'jpg'), data)
        self._write_tensorboard()
        self._print_current_errors()
        self.write_metrics(data)
        self.end_of_epoch(data, val_dataset, current_epoch, current_iteration)
        self.save_checkpoint(current_epoch, current_iteration)
        print('End debugging')

    def _set_custom_debug_parameter(self):
        r"""Set custom debug parame.
        """
        self.opt.logging_iter = 10
        self.opt.image_save_iter = 10

    def _write_custom_meters(self):
        r"""Dummy member function to be overloaded by the child class.
        In the child class, you can write down whatever you want to track.
        """
        pass

    @staticmethod
    def _write_to_meters(data, meters):
        r"""Write values to meters."""
        for key, value in data.items():
            meters[key].write(value)

    def _flush_meters(self, meters):
        r"""Flush all meters using the current iteration."""
        for meter in meters.values():
            meter.flush(self.current_iteration)

    def _pre_save_checkpoint(self):
        r"""Implement the things you want to do before saving a checkpoint.
        For example, you can compute the K-mean features (pix2pixHD) before
        saving the model weights to a checkpoint.
        """
        pass

    def save_checkpoint(self, current_epoch, current_iteration):
        r"""Save network weights, optimizer parameters, scheduler parameters
        to a checkpoint.
        """
        self._pre_save_checkpoint()
        # for distributed master design
        _save_checkpoint(self.opt, self.net_G_module, self.net_G_ema, self.opt_G, self.sch_G, current_epoch,
                         current_iteration)

    def load_checkpoint(self, opt, which_iter=None):
        if which_iter is not None:
            model_path = os.path.join(
                opt.logdir, '*_iteration_{:09}_checkpoint.pt'.format(which_iter))
            latest_checkpoint_path = glob.glob(model_path)
            assert len(latest_checkpoint_path) <= 1, "please check the saved model {}".format(
                model_path)
            if len(latest_checkpoint_path) == 0:
                current_epoch = 0
                current_iteration = 0
                print('No checkpoint found at iteration {}.'.format(which_iter))
                return current_epoch, current_iteration
            checkpoint_path = latest_checkpoint_path[0]

        elif os.path.exists(os.path.join(opt.logdir, 'latest_checkpoint.txt')):
            with open(os.path.join(opt.logdir, 'latest_checkpoint.txt'), 'r') as f:
                line = f.readlines()[0].replace('\n', '')
                checkpoint_path = os.path.join(opt.logdir, line.split(' ')[-1])
        else:
            current_epoch = 0
            current_iteration = 0
            print('No checkpoint found.')
            return current_epoch, current_iteration
        resume = opt.phase == 'train' and opt.resume
        current_epoch, current_iteration = self._load_checkpoint(
            checkpoint_path, resume)
        return current_epoch, current_iteration

    def _load_checkpoint(self, checkpoint_path, resume=True):
        current_epoch, current_iteration = 0, 0
        # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # self.net_G.load_state_dict(checkpoint['net_G'], strict=False)
        # self.net_G_ema.load_state_dict(checkpoint['net_G_ema'], strict=False)
        # print('load [net_G] and [net_G_ema] from {}'.format(checkpoint_path))
        # if self.opt.phase == 'train' and resume:
        #     # the checkpoint we provided does not contains
        #     # the parameters of the optimizer and schdule
        #     # because we train the model use another code
        #     # which does not save these parameters
        #     self.opt_G.load_state_dict(checkpoint['opt_G'])
        #     self.sch_G.load_state_dict(checkpoint['sch_G'])
        #     print('load optimizers and schdules from {}'.format(checkpoint_path))
        #
        # if resume or self.opt.phase == 'test':
        #     current_epoch = checkpoint['current_epoch']
        #     current_iteration = checkpoint['current_iteration']
        # else:
        #     current_epoch = 0
        #     current_iteration = 0
        # print('Done with loading the checkpoint.')
        return current_epoch, current_iteration

    def start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        data = self._start_of_iteration(data, current_iteration)
        data = to_cuda(data)
        self.current_iteration = current_iteration
        if not self.is_inference:
            self.net_G.train()
        self.start_iteration_time = time.time()
        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Things to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        # Update the learning rate policy for the generator if operating in the
        # iteration mode.
        if self.opt.gen_optimizer.lr_policy.iteration_mode:
            self.sch_G.step()
        # Accumulate time
        # torch.cuda.synchronize()
        self.elapsed_iteration_time += time.time() - self.start_iteration_time
        # Logging.
        if current_iteration % self.opt.logging_iter == 0:
            ave_t = self.elapsed_iteration_time / self.opt.logging_iter
            self.time_iteration = ave_t
            print('Iteration: {}, average iter time: '
                  '{:6f}.'.format(current_iteration, ave_t))
            self.elapsed_iteration_time = 0

            if getattr(self.opt, 'speed_benchmark', False):
                # Below code block only needed when analyzing computation
                # bottleneck.
                print('\tGenerator FWD time {:6f}'.format(
                    self.accu_gen_forw_iter_time / self.opt.logging_iter))
                print('\tGenerator LOS time {:6f}'.format(
                    self.accu_gen_loss_iter_time / self.opt.logging_iter))
                print('\tGenerator BCK time {:6f}'.format(
                    self.accu_gen_back_iter_time / self.opt.logging_iter))
                print('\tGenerator STP time {:6f}'.format(
                    self.accu_gen_step_iter_time / self.opt.logging_iter))
                print('\tGenerator AVG time {:6f}'.format(
                    self.accu_gen_avg_iter_time / self.opt.logging_iter))
                print('{:6f}'.format(ave_t))

                self.accu_gen_forw_iter_time = 0
                self.accu_gen_loss_iter_time = 0
                self.accu_gen_back_iter_time = 0
                self.accu_gen_step_iter_time = 0
                self.accu_gen_avg_iter_time = 0

        self._end_of_iteration(data, current_epoch, current_iteration)
        # Save everything to the checkpoint.
        if current_iteration >= self.opt.snapshot_save_start_iter and \
            current_iteration % self.opt.snapshot_save_iter == 0:
            self.save_image(self._get_save_path('image', 'jpg'), data)
            self.save_checkpoint(current_epoch, current_iteration)
            self.write_metrics(data)
        # Compute image to be saved.
        elif current_iteration % self.opt.image_save_iter == 0:
            self.save_image(self._get_save_path('image', 'jpg'), data)

        if current_iteration % self.opt.logging_iter == 0:
            self._write_tensorboard()
            self._print_current_errors()

    def _print_current_errors(self):
        epoch, iteration = self.current_epoch, self.current_iteration
        message = '(epoch: %d, iters: %d) ' % (epoch, iteration)
        for loss_name, losses in self.gen_losses.items():
            full_loss_name = 'gen_update' + '/' + loss_name
            message += '%s: %.3f ' % (full_loss_name, losses)

        print(message)
        log_name = os.path.join(self.opt.logdir, 'loss_log.txt')
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def end_of_epoch(self, data, val_dataset, current_epoch, current_iteration):
        r"""Things to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.

            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        # Update the learning rate policy for the generator if operating in the
        # epoch mode.
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if not self.opt.gen_optimizer.lr_policy.iteration_mode:
            self.sch_G.step()

        elapsed_epoch_time = time.time() - self.start_epoch_time
        # Logging.
        print('Epoch: {}, total time: {:6f}.'.format(current_epoch,
                                                     elapsed_epoch_time))
        self.time_epoch = elapsed_epoch_time
        self._end_of_epoch(data, current_epoch, current_iteration)
        # Save everything to the checkpoint.
        if current_epoch >= self.opt.snapshot_save_start_epoch and \
            current_epoch % self.opt.snapshot_save_epoch == 0:
            self.save_image(self._get_save_path('image', 'jpg'), data)
            self.save_checkpoint(current_epoch, current_iteration)
            self.write_metrics(data)
        if self.current_epoch % self.opt.eval_epoch == 0 and self.current_epoch >= self.opt.start_eval_epoch:
            self.eval(val_dataset)

    def eval(self, val_dataset):
        pass
        # output_dir = os.path.join(
        #     self.opt.logdir, 'evaluation',
        #     'epoch_{:05}_iteration_{:09}'.format(self.current_epoch, self.current_iteration)
        #     )
        # os.makedirs(output_dir, exist_ok=True)
        # lpips = self.test(val_dataset, output_dir, self.current_iteration)
        # self.write_data_tensorboard({'test_lpips': lpips.mean()},
        #                             self.current_epoch, self.current_iteration)

    def write_data_tensorboard(self, data, epoch, iteration):
        for name, value in data.items():
            full_name = 'eval/' + name
            if full_name not in self.meters.keys():
                # Create a new meter if it doesn't exist.
                self.meters[full_name] = Meter(full_name)
            self.meters[full_name].write(value)
            self.meters[full_name].flush(iteration)

    def save_image(self, path, data):
        r"""Compute visualization images and save them to the disk.

        Args:
            path (str): Location of the file.
            data (dict): Data used for the current iteration.
        """
        # important for distribute training
        if get_rank() != 0:
            return
        self.net_G.eval()
        vis_images = self._get_visualizations(data)
        if vis_images is not None:
            vis_images = (vis_images + 1) / 2
            print('Save output images to {}'.format(path))
            vis_images.clamp_(0, 1)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image_grid = torchvision.utils.make_grid(
                vis_images, nrow=1, padding=0, normalize=False)
            if self.opt.trainer.image_to_tensorboard:
                self.image_meter.write_image(image_grid, self.current_iteration)
            torchvision.utils.save_image(image_grid, path, nrow=1)

    def write_metrics(self, data):
        r"""Write metrics to the tensorboard."""
        if get_rank() != 0:
            return
        cur_metrics = self._compute_metrics(data, self.current_iteration)
        if cur_metrics is not None:
            if self.best_lpips is not None:
                self.best_lpips = min(self.best_lpips, cur_metrics['lpips'])
            else:
                self.best_lpips = cur_metrics['lpips']
            metric_dict = {
                'metric/lpips': cur_metrics['lpips'], 'metric/best_lpips': self.best_lpips
            }
            self._write_to_meters(metric_dict, self.meters)
            self._flush_meters(self.meters)
            if self.opt.trainer.hparam_to_tensorboard:
                add_hparams(self.hparam_dict, metric_dict)

    def _get_save_path(self, subdir, ext):
        r"""Get the image save path.

        Args:
            subdir (str): Sub-directory under the main directory for saving
                the outputs.
            ext (str): Filename extension for the image (e.g., jpg, png, ...).
        Return:
            (str): image filename to be used to save the visualization results.
        """
        subdir_path = os.path.join(self.opt.logdir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(
            subdir_path, 'epoch_{:05}_iteration_{:09}.{}'.format(
                self.current_epoch, self.current_iteration, ext))

    def _compute_metrics(self, data, current_iteration):
        r"""Return the evaluation result.
        """
        return None

    def _start_of_epoch(self, current_epoch):
        r"""Operations to do before starting an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        pass

    def _start_of_iteration(self, data, current_iteration):
        r"""Operations to do before starting an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current epoch number.
        Returns:
            (dict): Data used for the current iteration. They might be
                processed by the custom _start_of_iteration function.
        """
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Operations to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Operations to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _get_visualizations(self, data):
        r"""Compute visualization outputs.

        Args:
            data (dict): Data used for the current iteration.
        """
        return None

    def _init_loss(self, opt):
        r"""Every trainer should implement its own init loss function."""
        raise NotImplementedError

    def gen_forward(self, data):
        r"""Every trainer should implement its own generator forward."""
        raise NotImplementedError

    def test(self, data_loader, output_dir, current_iteration, test_limit=100):
        r"""Compute results images for a batch of input data and save the
        results in the specified folder.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch dataloader.
            output_dir (str): Target location for saving the output image.
        """
        raise NotImplementedError


@master_only
def _save_checkpoint(
    opt,
    net_G,
    net_G_ema,
    opt_G,
    sch_G,
    current_epoch,
    current_iteration
):
    latest_checkpoint_path = 'epoch_{:05}_iteration_{:09}_checkpoint.pt'.format(
        current_epoch, current_iteration)
    save_path = os.path.join(opt.logdir, latest_checkpoint_path)
    # drop the pretrained StyleGAN and Inversion part module
    net_G = net_G.state_dict()
    mini_net_G = {}
    for key in net_G:
        if 'generator' not in key:
            mini_net_G[key] = net_G[key]

    net_G_ema = net_G_ema.state_dict()
    mini_net_G_ema = {}
    for key in net_G_ema:
        if 'generator' not in key:
            mini_net_G_ema[key] = net_G_ema[key]
    del net_G, net_G_ema
    torch.save(
        {
            'net_G': mini_net_G,
            'net_G_ema': mini_net_G_ema,
            'opt_G': opt_G.state_dict(),
            'sch_G': sch_G.state_dict(),
            'current_epoch': current_epoch,
            'current_iteration': current_iteration,
        },
        save_path,
    )
    fn = os.path.join(opt.logdir, 'latest_checkpoint.txt')
    with open(fn, 'wt') as f:
        f.write('latest_checkpoint: %s' % latest_checkpoint_path)
    print('Save checkpoint to {}'.format(save_path))
    return save_path

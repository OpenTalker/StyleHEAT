import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.gan_loss import GANLoss
from loss.perceptual import PerceptualLoss
from torchvision.ops import roi_align
from utils.distributed import master_only_print as print
from utils.misc import to_cuda
from utils.trainer import accumulate
from utils.landmark import get_landmark_bbox
from utils.common import tensor2img
from trainers.base_trainer import BaseTrainer


class VideoWarperTrainer(BaseTrainer):

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
        super(VideoWarperTrainer, self).__init__(
            opt, net_G, net_G_ema, opt_G, sch_G,
            train_data_loader, val_data_loader
        )
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        self.visualization_buffer = {}
        self.metrics_buffer = {}

    def _init_loss(self, opt):
        self._assign_criteria(
            'perceptual_warp',
            PerceptualLoss(
                network=opt.trainer.vgg_param_warp.network,
                layers=opt.trainer.vgg_param_warp.layers,
                num_scales=getattr(opt.trainer.vgg_param_warp, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_warp, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_warp, 'style_to_perceptual', 0)
            ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_warp
        )
        self._assign_criteria(
            'perceptual_warp_middle',
            PerceptualLoss(
                network=opt.trainer.vgg_param_warp.network,
                layers=opt.trainer.vgg_param_warp.layers,
                num_scales=getattr(opt.trainer.vgg_param_warp, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_warp, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_warp, 'style_to_perceptual', 0)
            ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_warp_middle
        )

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data, if_optimize=True):
        self.opt_G.zero_grad()
        self.gen_losses = {}

        source_image = data['source_image']  # B, 3, 256, 256
        target_image = data['target_image']  # B, 3, 256, 256
        source_semantic = data['source_semantics']  # B, 73, 27
        target_semantic = data['target_semantics']  # B, 73, 27

        if random.randint(0, 1) == 1:
            source_mirror_image = torch.flip(source_image, dims=[3]).detach()
            input_image = torch.cat((source_image, source_mirror_image), 0)
        else:
            input_image = torch.cat((source_image, target_image), 0)

        gt_semantic = torch.cat((target_semantic, source_semantic), 0)
        gt_image = torch.cat((target_image, source_image), 0)
        # gt_lm = torch.cat((target_keypoint, source_keypoint), 0)

        output_dict = self.net_G(source_image=input_image, driven_3dmm=gt_semantic)
        fake_image = output_dict['final_image']

        # rec loss
        if self.opt.trainer.loss_weight.weight_perceptual_warp > 0:
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](fake_image, gt_image)

        if self.opt.trainer.loss_weight.weight_perceptual_warp_middle > 0:
            self.gen_losses["perceptual_warp_middle"] = \
                self.criteria['perceptual_warp_middle'](output_dict['w_source_image'], gt_image) + \
                self.criteria['perceptual_warp_middle'](output_dict['w_mirror_image'], gt_image)

        total_loss = torch.tensor(0.0).cuda()
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss
        if if_optimize:
            self.net_G.zero_grad()
            total_loss.backward()
            self.opt_G.step()
            accumulate(self.net_G_ema, self.net_G_module, self.accum)
        return total_loss.item()

    def _start_of_iteration(self, data, current_iteration):
        return data

    def inference(self, data):
        source_image = data['source_image']  # B, 3, 512, 512
        target_image = data['target_image']  # B, 3, 512, 512
        target_semantic = data['target_semantics']  # B, 73, 27

        input_image = source_image
        input_image[3:6] = torch.flip(input_image[3:6], dims=[3])
        gt_image = target_image

        self.net_G_ema.eval()
        output_dict = self.net_G_ema(source_image=input_image, driven_3dmm=target_semantic)

        self.visualization_buffer = {
            'input_image': input_image.detach(),
            'gt_image': gt_image.detach(),
            'fake_image': output_dict['final_image'].detach(),
            'w_source_image': output_dict['w_source_image'].detach(),
            'w_mirror_image': output_dict['w_mirror_image'].detach()
        }
        return

    def _get_visualizations(self, data):
        with torch.no_grad():
            self.inference(data)

            input_image = self.visualization_buffer['input_image']
            gt_image = self.visualization_buffer['gt_image']
            fake_image = self.visualization_buffer['fake_image']
            w_source_image = self.visualization_buffer['w_source_image']
            w_mirror_image = self.visualization_buffer['w_mirror_image']

            sample = torch.cat([input_image, gt_image, w_source_image, w_mirror_image, fake_image], 3)
            sample = torch.cat(torch.chunk(sample, sample.size(0), 0)[:6], 2)
        # Reset buffer
        self.visualization_buffer = {}
        return sample

    def test(self, data_loader, output_dir, current_iteration=-1, test_limit=100):
        output_dir = os.path.join(
            self.opt.logdir, 'evaluation',
            'epoch_{:03}_iteration_{:07}'.format(self.current_epoch, self.current_iteration)
            )
        os.makedirs(output_dir, exist_ok=True)
        self.net_G_ema.eval()

        total_loss = 0.0
        cnt = -1
        for it, data in enumerate(data_loader):
            if it >= test_limit:
                break
            cnt += 1
            data = to_cuda(data)
            with torch.no_grad():
                self.inference(data)
                loss = self.lpips(
                    self.visualization_buffer['fake_image'],
                    self.visualization_buffer['gt_image']
                ).mean()
            total_loss += loss
            if cnt % 10 == 0:
                self.save_image(os.path.join(output_dir, f'{it}.jpg'), data)
        total_loss /= cnt
        self.write_data_tensorboard({'test_lpips': total_loss},
                                    self.current_epoch, self.current_iteration)

    def _compute_metrics(self, data, current_iteration):
        metrics = {}
        with torch.no_grad():
            self.inference(data)
            metrics['lpips'] = self.lpips(
                self.visualization_buffer['fake_image'],
                self.visualization_buffer['gt_image']
            ).mean()
        return metrics

import os
import math
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


class VideoStyleHEATTrainer(BaseTrainer):

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
        super(VideoStyleHEATTrainer, self).__init__(
            opt, net_G, net_G_ema, opt_G, sch_G,
            train_data_loader, val_data_loader
        )
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        self.visualization_buffer = {}
        self.metrics_buffer = {}

    def _init_loss(self, opt):
        self._assign_criteria(
            'gan_loss',
            GANLoss().to('cuda'),
            opt.trainer.loss_weight.weight_gan_loss
        )
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
            'perceptual_final',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
            ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_final
        )
        self._assign_criteria(
            'perceptual_regular',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
            ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_regular
        )
        self._assign_criteria(
            'local_loss',
            nn.L1Loss(reduction='mean'),
            opt.trainer.loss_weight.weight_local_loss
        )

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data, if_optimize=True):
        self.opt_G.zero_grad()
        self.gen_losses = {}

        source_image = data['source_image']  # B, 3, 512, 512
        target_image = data['target_image']  # B, 3, 512, 512
        source_semantic = data['source_semantics']  # B, 73, 27
        target_semantic = data['target_semantics']  # B, 73, 27
        source_keypoint = data['source_keypoint']  # B, 68, 2
        target_keypoint = data['target_keypoint']  # B, 68, 2
        source_align = data['source_align']  # B, 3, 512, 512
        target_align = data['target_align']  # B, 3, 512, 512

        input_image = torch.cat((source_align, target_align), 0)
        gt_semantic = torch.cat((target_semantic, source_semantic), 0)
        gt_image = torch.cat((target_image, source_image), 0)
        gt_lm = torch.cat((target_keypoint, source_keypoint), 0)

        output_dict = self.net_G(input_image=input_image, driven_3dmm=gt_semantic)

        fake_image = output_dict['fake_image']
        # gt_img 512, landmark keypoint extracted at the size of 256
        gt_lm = gt_lm * 2  # NOTE, this is very important
        gt_bboxs_512 = get_landmark_bbox(gt_lm, scale=1)

        if self.current_iteration < 500:
            _lr = 0.2
        elif self.current_iteration < 2000:
            _lr = 0.05
        else:
            _lr = 0.005
        mask = torch.ones_like(fake_image) * _lr
        for box in gt_bboxs_512:
            for _i in range(box.shape[0]):
                lx, rx, ly, ry = box[_i]
                mask[_i, :, rx:ry, lx:ly] = 1

        if self.opt.trainer.loss_weight.weight_local_loss > 0:
            _i = torch.arange(gt_image.shape[0]).unsqueeze(1).cuda()

            gt_mouth_bbox = torch.cat([_i, gt_bboxs_512[0]], dim=1).float().cuda()
            gt_mouth = roi_align(gt_image, boxes=gt_mouth_bbox, output_size=120)  # gt_mouth: (2, 3, 120, 120)
            fake_mouth = roi_align(fake_image, boxes=gt_mouth_bbox, output_size=120)

            gt_l_eye_bbox = torch.cat([_i, gt_bboxs_512[1]], dim=1).float().cuda()
            gt_l_eye = roi_align(gt_image, boxes=gt_l_eye_bbox, output_size=80)
            fake_l_eye = roi_align(fake_image, boxes=gt_l_eye_bbox, output_size=80)

            gt_r_eye_bbox = torch.cat([_i, gt_bboxs_512[2]], dim=1).float().cuda()
            gt_r_eye = roi_align(gt_image, boxes=gt_r_eye_bbox, output_size=80)
            fake_r_eye = roi_align(fake_image, boxes=gt_r_eye_bbox, output_size=80)

            self.gen_losses["local_loss"] = \
                self.criteria['local_loss'](fake_mouth, gt_mouth) + \
                self.criteria['local_loss'](fake_l_eye, gt_l_eye) + \
                self.criteria['local_loss'](fake_r_eye, gt_r_eye)

        # rec loss
        if self.opt.trainer.loss_weight.weight_perceptual_warp > 0:
            warp_image = output_dict['video_warp_image']
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](warp_image, gt_image)

        if self.opt.trainer.loss_weight.weight_perceptual_final > 0:
            # add soft mask
            self.gen_losses["perceptual_final"] = \
                self.criteria['perceptual_final'](fake_image * mask, gt_image * mask)

        if self.opt.trainer.loss_weight.weight_perceptual_regular > 0:
            warp_image = output_dict['video_warp_image']
            regular_mask = 1 - mask
            self.gen_losses["perceptual_regular"] = \
                self.criteria['perceptual_regular'](fake_image * regular_mask, warp_image.detach() * regular_mask)

        if self.opt.trainer.loss_weight.weight_gan_loss > 0:
            # Note: for use the pre-trained discriminator, the batchsize are supposed to be times of 4
            self.gen_losses['gan_loss'] = self.criteria['gan_loss'](fake_image)

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
        source_align = data['source_align']  # B, 3, 512, 512

        input_image = source_align
        gt_image = target_image

        self.net_G_ema.eval()
        output_dict = self.net_G_ema(input_image=input_image, driven_3dmm=target_semantic)

        self.visualization_buffer = {
            'input_image': input_image.detach(),
            'gt_image': gt_image.detach(),
            'video_warp_image': output_dict['video_warp_image'].detach(),
            'fake_image': output_dict['fake_image'].detach()
        }
        return

    def _get_visualizations(self, data):
        with torch.no_grad():
            self.inference(data)

            input_image = self.visualization_buffer['input_image']
            gt_image = self.visualization_buffer['gt_image']
            video_warp_image = self.visualization_buffer['video_warp_image']
            fake_image = self.visualization_buffer['fake_image']

            sample = torch.cat([input_image, gt_image, video_warp_image, fake_image], 3)
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

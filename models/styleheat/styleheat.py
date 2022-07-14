import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flow_util
from utils.distributed import master_only_print as print
from models.hfgi.hfgi import HFGI
from models.styleheat.calibration_net import CalibrationNet
from models.styleheat.warper import AudioWarper, VideoWarper


class StyleHEAT(nn.Module):

    def __init__(self, opt, path_dic):
        super(StyleHEAT, self).__init__()

        self.opt = opt
        self.video_warper = VideoWarper()
        self.calibrator = CalibrationNet(
            out_size=64,  # check whether can enlarge the out size
            input_channel=512,  # refine feature channel is 512
            num_style_feat=512,
            channel_multiplier=2,
            # for stylegan decoder
            narrow=1
        )
        self.generator = HFGI()

        self.enable_audio = opt.enable_audio
        if self.enable_audio:
            self.audio_warper = AudioWarper()
            print('Enable audio driven.')

        self.load_checkpoint(opt, path_dic)

        self.frozen_params = ['video_warper']
        self.freeze_models()

    def freeze_models(self):
        for n in self.frozen_params:
            for p in self.__getattr__(n).parameters():
                p.requires_grad = False

    def load_checkpoint(self, opt, path_dic):
        self.generator.load_checkpoint(path_dic)
        print(f'Stage: {opt.mode}')
        if opt.mode == 'train_visual_refine':
            # Load from origin PIRender
            path = opt.visual_warper_path
            ckpt = torch.load(path, map_location='cpu')['net_G_ema']
            self.video_warper.load_state_dict(ckpt, strict=True)
            self.video_warper.eval()
            print(f'Load pre-trained VideoWarper [net_G_ema] from {opt.visual_warper_path} done')
        elif opt.mode == 'inference' or opt.mode == 'train_audio_refine':
            # Load from FreeStyler path
            path = opt.free_styler_path
            ckpt = torch.load(path, map_location='cpu')['net_G_ema']
            self.load_state_dict(ckpt, strict=False)  # should be full without StyleGAN
            self.eval()
            print(f'Load pre-trained StyleHEAT [net_G_ema] from {opt.free_styler_path} done')

            if opt.mode == 'train_audio_refine' and self.enable_audio:
                path = opt.audio_warper_path
                ckpt = torch.load(path, map_location='cpu')['net_G_ema']
                self.audio_warper.load_state_dict(ckpt, strict=True)
                self.audio_warper.eval()
                print(f'Load pre-trained AudioWarper from {path} done.')
        else:
            raise NotImplementedError

    def forward(self, input_image, driven_3dmm, driven_audio=None, inv_data=None, imsize=512):
        # Stage 1: Inversion
        if inv_data is None:
            with torch.no_grad():
                ix, wx, fx, inversion_condition = self.generator.inverse(input_image)
        else:
            # be careful about the batch case
            ix, wx, fx, inversion_condition = inv_data

        # Stage 2: Visual Warping
        video_output = self.video_warper(ix, driven_3dmm)  # Input: 256*256
        flow = video_output['flow_field']
        descriptor = video_output['descriptor']
        video_warping_condition = flow_util.convert_flow_to_deformation(flow)

        warping_condition = [video_warping_condition]

        fx_warp = flow_util.warp_image(fx, video_warping_condition)
        video_warp_img, _, _ = self.generator(
            [wx],
            warping_condition=warping_condition,
            inversion_condition=inversion_condition
        )
        video_warp_img = F.interpolate(video_warp_img, size=(imsize, imsize), mode="bilinear", align_corners=False)

        # Stage 3: Audio Warping
        if self.enable_audio:
            video_warp_img_256 = F.interpolate(video_warp_img, size=(256, 256), mode="bilinear", align_corners=False)
            flow = self.audio_warper(video_warp_img_256, driven_audio)['flow_field']  # Input: 256*256
            # TODO: trick flow: (B, 2, 64, 64) for inference only
            flow[:, :, :32] = 0

            audio_warping_condition = flow_util.convert_flow_to_deformation(flow)
            warping_condition.append(audio_warping_condition)

            fx_warp = flow_util.warp_image(fx_warp, audio_warping_condition)
            audio_warp_img, _, _ = self.generator(
                [wx],
                warping_condition=warping_condition,
                inversion_condition=inversion_condition
            )
            audio_warp_img = F.interpolate(audio_warp_img, size=(imsize, imsize), mode="bilinear", align_corners=False)
        else:
            audio_warp_img = None

        refining_condition = self.calibrator(fx_warp, descriptor)
        # refining_condition = self.calibrator(fx_warp)
        fake, _, _ = self.generator(
            [wx],
            f_condition=fx,
            refining_condition=refining_condition,
            warping_condition=warping_condition,
            inversion_condition=inversion_condition
        )
        fake = F.interpolate(fake, size=(imsize, imsize), mode="bilinear", align_corners=False)
        return {
            'fake_image': fake,
            'audio_warp_image': audio_warp_img,
            'video_warp_image': video_warp_img,
            'fx_warp': fx_warp
        }


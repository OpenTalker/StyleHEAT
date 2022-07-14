# Reference from: https://github.com/Tengfei-Wang/HFGI

import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.functional import interpolate
from models.e4e.e4e import E4eEncoder
from models.hfgi.backbone import ResidualEncoder, ResidualAligner
from models.stylegan2.model import Generator
from models.hfgi.editing import LatentEditor

from loss.iterative_inversion_loss import EmbeddingLossBuilder
from utils.common import tensor2img
from utils.distributed import master_only_print as print


class HFGI(nn.Module):

    def __init__(self):
        super(HFGI, self).__init__()

        self.warp_index = 7
        self.refine_index = 7
        # REFINE_MAP = {
        #     7: (64, 512),
        #     9: (128, 256),
        #     11: (256, 128)
        # }
        # print(f'[NOTE]: Warp Index of StyleGAN: {self.warp_index}, Refine Index of StyleGAN: {self.refine_index}')

        self.e4e_encoder = E4eEncoder(latent_avg=None)
        self.hfgi_aligner = ResidualAligner()
        self.hfgi_encoder = ResidualEncoder()
        self.stylegan = Generator(
            size=1024,
            style_dim=512,
            n_mlp=8,
            channel_multiplier=2,
        )

        self.iteration_loss = EmbeddingLossBuilder()
        self.latent_editor = LatentEditor()

    def load_checkpoint(self, path_dic):
        # e4e
        e4e_path = path_dic['e4e']
        ckpt = torch.load(e4e_path, map_location='cpu')
        self.e4e_encoder.latent_avg = ckpt['latent_avg'].cuda()
        self.e4e_encoder.encoder.load_state_dict(ckpt['encoder'])
        self.e4e_encoder.eval()
        print(f'Load pre-trained e4e Encoder from {e4e_path} done.')

        # HFGI
        hfgi_path = path_dic['hfgi']
        ckpt = torch.load(hfgi_path, map_location='cpu')
        self.hfgi_aligner.load_state_dict(ckpt['aligner'])
        self.hfgi_aligner.eval()
        self.hfgi_encoder.load_state_dict(ckpt['encoder'])
        self.hfgi_encoder.eval()
        print(f'Load pre-trained hfgi encoder from {hfgi_path} done.')

        # stylegan
        stylegan_path = path_dic['stylegan2']
        ckpt = torch.load(stylegan_path, map_location='cpu')
        self.stylegan.load_state_dict(ckpt)
        self.stylegan.eval()
        print(f'Load pre-trained StyleGAN2 from {stylegan_path} done.')
        # E.g. model([codes], input_is_latent=True, randomize_noise=False, return_latents=True)

        # editing
        self.latent_editor.load(path_dic)

    def encode(self, x):
        # e4e rough inversion
        x = interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        wx = self.e4e_encoder(x)
        ix, wx, _ = self.stylegan([wx])

        ix = interpolate(torch.clamp(ix, -1., 1.), size=(256, 256), mode='bilinear', align_corners=False)
        res = (x - ix).detach()
        return wx, ix, res

    def decode(self, wx, ix, res, output_size=256):
        # align the distortion map
        res_align = self.hfgi_aligner(torch.cat((res, ix), 1))
        inversion_condition = self.hfgi_encoder(res_align)
        ix, wx, fx = self.stylegan([wx], inversion_condition=inversion_condition)
        ix = interpolate(torch.clamp(ix, -1., 1.), size=(output_size, output_size), mode='bilinear',
                         align_corners=False)
        return ix, wx, fx, inversion_condition

    def inverse(self, x):
        # Input 256, 256
        # Output 256, 256
        wx, ix, res = self.encode(x)
        ix, wx, fx, inversion_condition = self.decode(wx, ix, res)
        return ix, wx, fx, inversion_condition

    def edit(self, x, factor=2, choice='age', wx=None, res=None, output_size=256):
        if wx is None or res is None:
            wx, ix, res = self.encode(x)
        # edit
        wx_edit = self.latent_editor.edit_style_code(wx, factor, choice)
        ix_edit, _, _ = self.stylegan([wx_edit])
        ix_edit = interpolate(torch.clamp(ix_edit, -1., 1.), size=(256, 256), mode='bilinear', align_corners=False)
        ix_edit, wx_edit, fx_edit, inversion_condition = self.decode(wx_edit, ix_edit, res, output_size)
        return ix_edit, wx_edit, fx_edit, inversion_condition

    def forward(self, wx, inversion_condition=None, refining_condition=None, warping_condition=None,
                f_condition=None):
        # output 1024*1024
        ix, wx, fx = self.stylegan(
            wx,
            f_condition=f_condition,
            inversion_condition=inversion_condition,
            refining_condition=refining_condition,
            warping_condition=warping_condition
        )
        return ix, wx, fx

    def e4e_inverse(self, x):
        # Input 256, 256
        # Output 256, 256
        wx, ix, res = self.encode(x)
        ix = interpolate(torch.clamp(ix, -1., 1.), size=(256, 256), mode='bilinear', align_corners=False)
        return ix, wx

    def optimize_inverse(self, x, save_path=None):
        # Borrowed from Barbershop for f-space inversion, optimize f and w jointly
        # W+ code (1, 18, 512)
        # F code (512, 64, 64) (After Ada)
        x = interpolate(x, (1024, 1024), mode='bilinear', align_corners=False)
        x_256 = interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        # First use encoder to achieve the rough f & s
        ix, wx, fx, inversion_condition = self.inverse(x_256)

        f_init = fx.clone().detach()
        # w_latent = wx  # torch.randn(1, 18, 512).cuda()
        # f_latent = fx  # torch.randn(1, 512, 64, 64).cuda()

        # Setup optimizer of f & w latent
        f_latent = fx.clone().detach().requires_grad_(True)
        w_latent = []
        start_index = 9
        for i in range(18):
            temp = wx[0, i].clone().detach()
            if i < start_index:
                temp.requires_grad = False
            else:
                temp.requires_grad = True
            w_latent.append(temp)
        optimizer_fs = torch.optim.Adam(w_latent[start_index:] + [f_latent], lr=0.01)

        # Iteratively optimization to increase the inversion performance
        temp_latent = None
        for _ in range(250):
            optimizer_fs.zero_grad()
            temp_latent = torch.stack(w_latent).unsqueeze(0)
            ix, _ = self.stylegan([temp_latent], return_latents=False, f_condition=f_latent)
            loss, loss_dic = self.iteration_loss(ix, x, temp_latent, f_latent, f_init)
            loss.backward()
            optimizer_fs.step()

        if save_path is not None:
            self.save_fs_latent(save_path, ix, temp_latent, f_latent)
        ix = interpolate(ix, (256, 256), mode='bilinear', align_corners=False)
        # print(ix.shape, temp_latent.shape, f_latent.shape)
        return ix, temp_latent, f_latent

    def save_fs_latent(self, save_path, image, w_latent, f_latent):
        basename = os.path.basename(save_path).split('.')[0]
        dirname = os.path.dirname(save_path)
        if basename == '':
            basename = 'i_inversion'
        latent_path = os.path.join(dirname, f'{basename}.npz')
        image_path = os.path.join(dirname, f'{basename}.jpg')
        tensor2img(image).save(image_path)
        np.savez(
            latent_path,
            w_latent=w_latent.detach().cpu().numpy(),
            f_latent=f_latent.detach().cpu().numpy()
        )

    def load_fs_latent(self, latent_path):
        latent_dic = np.load(latent_path)
        w_latent = torch.from_numpy(latent_dic['w_latent']).cuda()
        f_latent = torch.from_numpy(latent_dic['f_latent']).cuda()

        ix, _ = self.stylegan([w_latent], return_latents=False, f_feature=f_latent)
        ix = interpolate(ix, (256, 256), mode='bilinear', align_corners=False)
        return ix, w_latent, f_latent

    def random_code(self, factor=0.1):
        mean_code = self.e4e_encoder.latent_avg
        random_noise = torch.randn_like(mean_code).clamp(-factor, factor)
        return mean_code + random_noise

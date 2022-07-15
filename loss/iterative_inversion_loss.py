# borrowed from https://github.com/ZPdesu/Barbershop/losses/embedding_loss.py

import numpy as np
import torch
import torch.nn as nn

from loss.perceptual import PerceptualLoss
from configs.path import PRETRAINED_MODELS_PATH


class EmbeddingLossBuilder(nn.Module):
    def __init__(self):
        super(EmbeddingLossBuilder, self).__init__()
        # self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.perceptual_lambda, 'percep']]
        # perceptualual loss
        self.l2 = torch.nn.MSELoss()
        self.perceptual = PerceptualLoss(
                network='vgg19',
                layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                num_scales=4,
                use_style_loss=True,
                weight_style_to_perceptual=250
            ).to('cuda')
        # for p_norm loss
        self.load_PCA_model()

    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)

    def _loss_lpips(self, gen_im, ref_im, **kwargs):
        return self.perceptual(gen_im, ref_im)

    def forward(self, fake_image, gt_image, latent_in, latent_F, F_init):
        loss = 0
        loss_dic = {}
        loss_l2 = self._loss_l2(fake_image, gt_image) * 1.0
        loss_dic['l2'] = loss_l2
        loss += loss_l2

        # TODO maybe downsample is not necessary
        fake_image_256 = nn.functional.interpolate(fake_image, (256, 256), mode='bilinear', align_corners=False)
        gt_image_256 = nn.functional.interpolate(gt_image, (256, 256), mode='bilinear', align_corners=False)
        loss_lpips = self._loss_lpips(fake_image_256, gt_image_256) * 1.0
        loss_dic['lpips'] = loss_lpips
        loss += loss_lpips

        p_norm_loss = self.cal_p_norm_loss(latent_in)  # done
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        l_F = self.cal_l_F(latent_F, F_init)
        loss_dic['l_F'] = l_F
        loss += l_F
        return loss, loss_dic

    def cal_l_F(self, latent_F, F_init):
        self.l_F_lambda = 0.1
        return self.l_F_lambda * (latent_F - F_init).pow(2).mean()

    def cal_p_norm_loss(self, latent_in):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = self.p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss

    def load_PCA_model(self):
        device = 'cuda'
        PCA_path = PRETRAINED_MODELS_PATH['FFHQ_PCA']

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)
        self.p_norm_lambda = 0.001

    # def build_PCA_model(self, PCA_path):
    #     with torch.no_grad():
    #         latent = torch.randn((1000000, 512), dtype=torch.float32)
    #         # latent = torch.randn((10000, 512), dtype=torch.float32)
    #         self.generator.style.cpu()
    #         pulse_space = torch.nn.LeakyReLU(5)(self.generator.style(latent)).numpy()
    #         self.generator.style.to(self.opts.device)
    #
    #     from utils.PCA_utils import IPCAEstimator
    #
    #     transformer = IPCAEstimator(512)
    #     X_mean = pulse_space.mean(0)
    #     transformer.fit(pulse_space - X_mean)
    #     X_comp, X_stdev, X_var_ratio = transformer.get_components()
    #     np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)

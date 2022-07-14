import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.losses import build_loss
from configs.path import PRETRAINED_MODELS_PATH
from models.stylegan2.model import Discriminator
from utils.distributed import master_only_print as print


class GANLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.net_d = Discriminator(size=256, channel_multiplier=2)
        self.net_d = self.net_d.to('cuda')

        print('Load pre-trained StyleGAN2 256 discriminator done.')
        pretrained_path = PRETRAINED_MODELS_PATH['discriminator']
        ckpt = torch.load(pretrained_path, map_location='cpu')
        self.net_d.load_state_dict(ckpt)
        self.net_d.eval()

        gan_opt = {
            'type': 'GANLoss',
            'gan_type': 'wgan_softplus',
            'loss_weight': 1e-1
        }
        self.gan_loss = build_loss(gan_opt).to('cuda')

    def forward(self, fake_image):
        # Note: for use the pre-trained discriminator, the batchsize are supposed to be times of 4
        fake_image = F.interpolate(fake_image, (256, 256), mode='bilinear', align_corners=False)
        fake_g_pred = self.net_d(fake_image)
        loss = self.gan_loss(fake_g_pred, target_is_real=False, is_disc=False)
        return loss

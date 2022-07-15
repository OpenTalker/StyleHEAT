import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flow_util
from models.styleheat.base_function import LayerNorm2d, ADAINHourglass


class VideoWarper(nn.Module):
    def __init__(
        self
    ):
        super(VideoWarper, self).__init__()
        self.mapping_net = MappingNet(
            coeff_nc=73,
            descriptor_nc=256,
            layer=3
        )
        self.warping_net = WarpingNet(
            encoder_layer=5,
            decoder_layer=3,
            base_nc=32,
            image_nc=3,
            descriptor_nc=256,
            max_nc=256,
            use_spect=False
        )

    def forward(
        self,
        input_image,
        driving_source
    ):
        """
        :param input_image:
        :param driving_source:
        :return: output: dict: {'warp_image', 'flow_field', 'descriptor'}
        """
        descriptor = self.mapping_net(driving_source)
        output = self.warping_net(input_image, descriptor)
        output['descriptor'] = descriptor
        return output


class AudioWarper(nn.Module):

    def __init__(self):
        super(AudioWarper, self).__init__()
        self.audio_encoder = MappingNet(
            coeff_nc=80,
            descriptor_nc=256,
            layer=3
        )
        self.warpping_net = WarpingNet(
            encoder_layer=5,
            decoder_layer=3,
            base_nc=32,
            image_nc=3,
            descriptor_nc=256,
            max_nc=256,
            use_spect=False
        )

    def forward(
        self,
        input_image,
        driving_source
    ):
        descriptor = self.audio_encoder(driving_source)
        # print(f'descritor.shape: {descriptor.shape}')
        output = self.warpping_net(input_image, descriptor)
        output['descriptor'] = descriptor
        return output


class MappingNet(nn.Module):

    def __init__(self, coeff_nc, descriptor_nc, layer):
        super(MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:, :, 3:-3]
        out = self.pooling(out)
        return out


class WarpingNet(nn.Module):

    def __init__(
        self,
        image_nc,
        descriptor_nc,
        base_nc,
        max_nc,
        encoder_layer,
        decoder_layer,
        use_spect
    ):
        super(WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'nonlinearity': nonlinearity, 'use_spect': use_spect}

        self.descriptor_nc = descriptor_nc
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc,
                                        max_nc, encoder_layer, decoder_layer, **kwargs)

        self.flow_out = nn.Sequential(norm_layer(self.hourglass.output_nc),
                                      nonlinearity,
                                      nn.Conv2d(self.hourglass.output_nc, 2, kernel_size=7, stride=1, padding=3))

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        final_output = {}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)

        deformation = flow_util.convert_flow_to_deformation(final_output['flow_field'])
        final_output['warp_image'] = flow_util.warp_image(input_image, deformation)
        return final_output


def test_audio_warper():
    model = AudioWarper().cuda()
    img = torch.randn(2, 3, 256, 256).cuda()
    wav = torch.randn(2, 80, 32).cuda()  # 2, 5,
    output = model(img, wav)
    print(output['flow_field'].shape)
    print(output['warp_image'].shape)


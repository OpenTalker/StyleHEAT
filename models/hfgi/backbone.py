import torch
import torch.nn as nn
from models.stylegan2.model import EqualConv2d, ScaledLeakyReLU
from models.e4e.helpers import bottleneck_IR


# Consultation encoder
class ResidualEncoder(nn.Module):
    def __init__(self, opts=None):
        super(ResidualEncoder, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.PReLU(32))

        self.conv_layer2 = nn.Sequential(*[bottleneck_IR(32,48,2), bottleneck_IR(48,48,1), bottleneck_IR(48,48,1)])

        self.conv_layer3 = nn.Sequential(*[bottleneck_IR(48,64,2), bottleneck_IR(64,64,1), bottleneck_IR(64,64,1)])

        self.condition_scale3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))

        self.condition_shift3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = nn.functional.interpolate(scale, size=(64,64) , mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = nn.functional.interpolate(shift, size=(64,64) , mode='bilinear')
        conditions.append(shift.clone())
        return conditions


# ADA
class ResidualAligner(nn.Module):
    def __init__(self, opts=None):
        super(ResidualAligner, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(6, 16, (3, 3), 1, 1, bias=False),
                                      nn.BatchNorm2d(16),
                                      nn.PReLU(16))

        self.conv_layer2 = nn.Sequential(*[bottleneck_IR(16, 32, 2), bottleneck_IR(32, 32, 1), bottleneck_IR(32, 32, 1)])
        self.conv_layer3 = nn.Sequential(*[bottleneck_IR(32, 48, 2), bottleneck_IR(48, 48, 1), bottleneck_IR(48, 48, 1)])
        self.conv_layer4 = nn.Sequential(*[bottleneck_IR(48, 64, 2), bottleneck_IR(64, 64, 1), bottleneck_IR(64, 64, 1)])

        self.dconv_layer1 = nn.Sequential(*[bottleneck_IR(112, 64, 1), bottleneck_IR(64, 32, 1), bottleneck_IR(32, 32, 1)])
        self.dconv_layer2 = nn.Sequential(*[bottleneck_IR(64, 32, 1), bottleneck_IR(32, 16, 1), bottleneck_IR(16, 16, 1)])
        self.dconv_layer3 = nn.Sequential(*[bottleneck_IR(32, 16, 1), bottleneck_IR(16, 3, 1), bottleneck_IR(3, 3, 1)])

    def forward(self, x):
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)
        feat4 = self.conv_layer4(feat3)

        feat4 = nn.functional.interpolate(feat4, size=(64, 64), mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat4, feat3), 1))
        dfea1 = nn.functional.interpolate(dfea1, size=(128, 128), mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat((dfea1, feat2), 1))
        dfea2 = nn.functional.interpolate(dfea2, size=(256, 256), mode='bilinear')
        dfea3 = self.dconv_layer3(torch.cat((dfea2, feat1), 1))

        res_aligned = dfea3
        return res_aligned

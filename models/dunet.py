import math
import torch
import torch.nn.functional as F
from torch import nn
from models import resnet
from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
import torch.utils.model_zoo as model_zoo
from itertools import chain

'''
adapted from https://github.com/hali1122/DUpsampling/blob/master/models/dunet.py
'''


class DUpsampling(nn.Module):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2, **kwargs):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 1, bias=False)

    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()

        # N, C, H, W --> N, W, H, C
        x = x.permute(0, 3, 2, 1).contiguous()

        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)

        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()

        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor))

        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = x.permute(0, 3, 1, 2)

        return x
    
''' 
-> ResNet BackBone
'''

class ResNet(nn.Module):
    def __init__(self, in_channels=3, backbone='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        # if not pretrained or in_channels != 3:
        #     self.layer0 = nn.Sequential(
        #         nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #     )
        #     initialize_weights(self.layer0)
        # else:
        #     self.layer0 = nn.Sequential(*list(model.children())[:4])
        layer0 = list(model.children())[:4]
        self.conv1 = layer0[0]
        self.bn1 = layer0[1]
        self.relu1 = layer0[2]
        self.maxpool1 = layer0[3]
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(self.relu1(self.bn1(x)))
        x_13 = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_46 = x
        x = self.layer4(x)
        x_13 = F.interpolate(x_13, [x_46.size()[2], x_46.size()[3]], mode='bilinear', align_corners=True)
        low_level_features = torch.cat((x_13, x_46), dim=1)
        return x, low_level_features

class _DUHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DUHead, self).__init__()
        self.fuse = FeatureFused(norm_layer=norm_layer, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(True)
        )

    def forward(self, x, low_level_features):
        fused_feature = self.fuse(x, low_level_features)
        out = self.block(fused_feature)
        return out

class FeatureFused(nn.Module):
    """Module for fused features"""

    def __init__(self, inter_channels=512, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(1024, inter_channels, 1, bias=False),
        #     norm_layer(inter_channels),
        #     nn.ReLU(True)
        # )

    def forward(self, x, low_level_features):
        size = low_level_features.size()[2:]
        x = self.conv2(F.interpolate(x, size, mode='bilinear', align_corners=True))
        fused_feature = torch.cat([x, low_level_features], dim=1)
        return fused_feature


class DUNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, backbone='resnet50', **kwargs):
        super(DUNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.encoder = ResNet()
        self.head = _DUHead(1664, **kwargs)
        self.dupsample = DUpsampling(256, num_classes, scale_factor=16, **kwargs)
        # self.decoder = Decoder(num_classes)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x, x_low = self.encoder(x)
        x = self.head(x, x_low)
        print('preup', x.size())
        x = self.dupsample(x)
        print('out', x.size())
        # I think this is breaking stuff
        x = x[:, :, :input_size[0], :input_size[1]]

        return x
    def get_backbone_params(self):
        return chain(self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.head.parameters(), self.dupsample.parameters())
    

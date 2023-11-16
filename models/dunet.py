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
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)

        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)

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

class Decoder(nn.Module):
    def __init__(self, num_class, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1152, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(2096, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1)

        self.dupsample = DUpsampling(256, 32, num_class=21)
        self._init_weight()
        self.T = torch.nn.Parameter(torch.Tensor([1.00]))

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        low_level_feature = F.interpolate(low_level_feature, [x.size()[2], x.size()[3]], mode='bilinear', align_corners=True)
        x_4_cat = torch.cat((x, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        out = self.dupsample(x_4_cat)
        out = out / self.T
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DUNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True, backbone='resnet50', **_):
        super(DUNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.encoder = ResNet()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x, x_low = self.encoder(x)
        x = self.decoder(x, x_low)
        x = x[:, :, :input_size[0], :input_size[1]]

        return x
    def get_backbone_params(self):
        return chain(self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.decoder.parameters())

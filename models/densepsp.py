import math
import torch
import torch.nn.functional as F
from torch import nn
from models import resnet
from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain

''' 
-> ResNet BackBone
'''

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet50', pretrained=True, dialated=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if dialated:
            if output_stride == 16:
                s3, s4, d3, d4 = (2, 1, 1, 2)
            elif output_stride == 8:
                s3, s4, d3, d4 = (1, 1, 2, 4)

            if output_stride == 8:
                for n, m in self.layer3.named_modules():
                    if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                        m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                    elif 'conv2' in n:
                        m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                    elif 'downsample.0' in n:
                        m.stride = (s3, s3)

            for n, m in self.layer4.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
                elif 'downsample.0' in n:
                    m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        return x, low_level_features, x_aux


class _DensePSPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, bin_sz,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DensePSPConv, self).__init__()
        '''
        use depthwise with 
            self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1, groups=in_channels)),
            may need to assert out_channels // in_channels
        use depthwise seperable with 
            self.add_module('conv1', SeparableConv2d(in_channels, inter_channels, 1))
        '''
        # keep these in line with the original psp for now
        # self.add_module('conv1', SeparableConv2d(in_channels, inter_channels, 1)),
        # self.add_module('bn1', norm_layer(inter_channels)),
        # self.add_module('relu1', nn.ReLU(True)),
        self.add_module('aap1', nn.AdaptiveAvgPool2d(output_size=bin_sz)),
        # can try to dialate this layer later
        self.add_module('conv2', SeparableConv2d(in_channels, out_channels, 1))
        self.add_module('bn2', norm_layer(out_channels)),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        features = F.interpolate(super(_DensePSPConv, self).forward(x), size=(h, w), mode='bilinear', align_corners=True)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features

class _DensePSPModule(nn.Module):
    def __init__(self, in_channels, inter_channels, bin_sizes, norm_layer):
        super(_DensePSPModule, self).__init__()
        # out
        stage_out = in_channels // len(bin_sizes)
        out_channels = in_channels // (len(bin_sizes))

        # stages = {psp_1,psp_2,psp_3,psp_6}
        '''
        OLD - tried to set up stages with list comprehension
        self.stages = nn.ModuleList([self._make_stages(in_channels + in_channels*i, stage_out, b_s, norm_layer)
                             for i, b_s in enumerate(bin_sizes)])
        '''
        '''
        TODO - check new List comp
        self.stages = nn.ModuleList([_DensePSPConv(in_channels + in_channels*i, inter_channels, stage_out, b_s, norm_layer)
                     for i, b_s in enumerate(bin_sizes)])
        '''
        #notes for 11-05_00-26: used bin sizes 2,3,6,9
        self.psp_2 = _DensePSPConv(in_channels, inter_channels, stage_out, 1, 0.1,
                                     norm_layer)
        self.psp_3 = _DensePSPConv(in_channels + stage_out * 1, inter_channels, stage_out, 2, 0.1,
                                     norm_layer)
        self.psp_6 = _DensePSPConv(in_channels + stage_out * 2, inter_channels, stage_out, 3, 0.1,
                                      norm_layer)
        self.psp_9 = _DensePSPConv(in_channels + stage_out * 3, inter_channels, stage_out, 6, 0.1,
                                      norm_layer)
        #TODO add a 12 and see what happens


        self.bottleneck = nn.Sequential(
            SeparableConv2d(in_channels + (stage_out * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    '''
    OLD - Tried to set up stages with list comprehensions
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        for psp_bin in self.stages:
            bin_out = F.interpolate(psp_bin(features), size=(h, w), mode='bilinear', align_corners=True)
            features = torch.cat((bin_out, features), dim=1)
            pyramids.append(bin_out)
        output = self.bottleneck(torch.cat(pyramids, dim=1))
    '''
    def forward(self, x):
        psp2 = self.psp_2(x)
        x = torch.cat([psp2, x], dim=1)

        psp3 = self.psp_3(x)
        x = torch.cat([psp3, x], dim=1)

        psp6 = self.psp_6(x)
        x = torch.cat([psp6, x], dim=1)

        psp9 = self.psp_9(x)
        x = torch.cat([psp9, x], dim=1)

        x = self.bottleneck(x)
        return x

class DensePSP(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='dilated_resnet50', inter_channels=256,
                 pretrained=True, use_aux=True, freeze_bn=False,
                 freeze_backbone=False, output_stride=8):
        super(DensePSP, self).__init__()
        norm_layer = nn.BatchNorm2d
        assert backbone == 'dilated_resnet50' or backbone == 'resnet50'
        self.dilated = (backbone == 'dilated_resnet50')
        self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained, dialated=self.dilated)

        m_out_sz = 2048
        self.use_aux = use_aux

        # note for depthwise m_out_sz // inter_channels
        self.master_branch = nn.Sequential(
            _DensePSPModule(m_out_sz, inter_channels=inter_channels, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),

            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            # can replace this with depthwise
            nn.Conv2d(m_out_sz // 2, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x, low_level_features, x_aux = self.backbone(x)
        x = self.master_branch(x)

        x = F.interpolate(x, size=input_size, mode='bilinear')
        x = x[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return x, aux
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
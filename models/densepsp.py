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
class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet50', pretrained=True, dilated=True, hdc=False, hdc_dilation_bigger=False):
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

        if hdc:
            d_res4b = []
            if hdc_dilation_bigger:
                d_res4b.extend([1, 2, 5, 9]*5 + [1, 2, 5])
                d_res5b = [5, 9, 17]
            else:
                # Dialtion-RF
                d_res4b.extend([1, 2, 3]*7 + [2, 2])
                d_res5b = [3, 4, 5]

            if output_stride == 8:
                l_index = 0
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        d = d_res4b[l_index]
                        m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                        l_index += 1
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)

            l_index = 0
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    d = d_res5b[l_index]
                    m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                    l_index += 1
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        elif dilated:
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
        x_13 = x
        x = self.layer1(x)
        # low_level_features = x
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        #1024+64 features -> 1088 low level feauture
        x_13 = F.interpolate(x_13, [x_aux.size()[2], x_aux.size()[3]], mode='bilinear', align_corners=True)
        low_level_features = torch.cat((x_13, x_aux), dim=1)
        # assert False
        return x, low_level_features, x_aux

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

class DensePSPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, bin_sz,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, pool_layer=nn.AdaptiveAvgPool2d, up_mode='bilinear'):
        super(DensePSPConv, self).__init__()
        self.add_module('pool', pool_layer(output_size=bin_sz))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, 1))
        self.add_module('bn', norm_layer(out_channels))
        self.add_module('relu', nn.ReLU(True))
        self.up_mode = up_mode

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        if self.up_mode == 'bilinear':
            features = F.interpolate(super(DensePSPConv, self).forward(x), size=(h, w), mode=self.up_mode, align_corners=True)
        else:
            features = F.interpolate(super(DensePSPConv, self).forward(x), size=(h, w), mode=self.up_mode)
        return features

class _DensePSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sz,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, pool_layer=nn.AdaptiveMaxPool2d, up_mode='bilinear', bin_increase=1,
                 norm_kwargs=None):
        super(_DensePSPStage, self).__init__()
        '''
        input -> aMaxPool(x) + 
        '''
        self.pool1 = nn.AdaptiveMaxPool2d(output_size=bin_sz+bin_increase)
        self.pool2 = pool_layer(output_size=bin_sz)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                  norm_layer(out_channels),
                                  nn.ReLU(inplace=True))
        self.up_mode = up_mode

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        x = self.pool1(x)
        if self.up_mode == 'bilinear':
            x = F.interpolate(x, size=(h, w), mode=self.up_mode, align_corners=True)
        else:
            x = F.interpolate(x, size=(h, w), mode=self.up_mode)
        x = self.pool2(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x
    

# TODO - 16, 32, 64, 128 added maps
# TODO - check global vs maxp for agregated feat
class _DensePSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer=nn.BatchNorm2d,
                 pool_layer=nn.AdaptiveMaxPool2d, up_mode='bilinear', bin_increase=1):
        super(_DensePSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        assert bin_sizes == [1,2,3,6]
        bin1 = DensePSPConv(in_channels, out_channels, bin_sizes[0], norm_layer=norm_layer)
        bin2 = _DensePSPStage(in_channels, out_channels, bin_sizes[1],
                              norm_layer=norm_layer, pool_layer=pool_layer, up_mode=up_mode, bin_increase=bin_increase)
        bin3 = _DensePSPStage(in_channels, out_channels, bin_sizes[2],
                              norm_layer=norm_layer, pool_layer=pool_layer, up_mode=up_mode, bin_increase=bin_increase)
        bin6 = _DensePSPStage(in_channels, out_channels, bin_sizes[3],
                              norm_layer=norm_layer, pool_layer=pool_layer, up_mode=up_mode, bin_increase=bin_increase)
        self.stages = nn.ModuleList([bin1, bin2, bin3, bin6])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), 512, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([stage(features) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class DensePSP(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', inter_channels=64, pool_layer='average',
                  up_mode='bilinear', bin_sizes=[1,2,3,6], pretrained=True, use_aux=True, freeze_bn=False,
                 dilated=True, output_stride=16, hdc=False, hdc_dilation_bigger=False, freeze_backbone=False):
        super(DensePSP, self).__init__()
        norm_layer = nn.BatchNorm2d
        assert pool_layer in ['average', 'max']
        assert up_mode in ['nearest', 'bilinear']
        self.backbone = ResNet(in_channels=in_channels, dilated=dilated, output_stride=output_stride,
                               pretrained=pretrained, backbone=backbone, hdc=hdc, hdc_dilation_bigger=hdc_dilation_bigger)
        pool_layer = nn.AdaptiveAvgPool2d if pool_layer == 'average' else nn.AdaptiveMaxPool2d
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux

        self.master_branch = nn.Sequential(
            _DensePSPModule(m_out_sz, bin_sizes, norm_layer=norm_layer, pool_layer=pool_layer,up_mode=up_mode),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz//2, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x, low_level_features, x_aux = self.backbone(x)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        # output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            # aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.backbone.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

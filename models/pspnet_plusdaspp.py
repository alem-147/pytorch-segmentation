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

#does not convolve at the end with the bottleneck
class _PSPModuleMaps(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModuleMaps, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = []
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        return torch.cat(pyramids, dim=1)


'''
Need to save memory on my 12 GB machine
'''
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, padding=1,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        # if dilation > kernel_size // 2:
        #     padding = dilation
        # else:
        #     padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        '''
        use depthwise with 
            self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1, groups=in_channels)),
        use depthwise seperable with 
            self.add_module('conv1', SeparableConv2d(in_channels, inter_channels, 1))
            dont have bn or relu afterwards
        '''
        self.add_module('conv1', SeparableConv2d(in_channels, inter_channels, 1, padding=0)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', SeparableConv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)

        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        return x


class _PSPPlusModule(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, bin_sizes, norm_layer):
        super(_PSPPlusModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_psp_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])
        self.dense_aspp_block = _DenseASPPBlock(in_channels, inter_channels1, inter_channels2, norm_layer)

        # self.psp_separable = nn.Sequential(
        #     SeparableConv2d(in_channels + out_channels * len(bin_sizes), 2048),
        #     norm_layer(2048),
        #     nn.ReLU(True),
        #     SeparableConv2d(2048, 1024),
        #     norm_layer(1024),
        #     nn.ReLU(True)
        # )

        self.daspp_separable = nn.Sequential(
            SeparableConv2d(in_channels + inter_channels2 * 5, 1024),
            norm_layer(1024),
            nn.ReLU(True),
            SeparableConv2d(1024, 512),
            norm_layer(512),
            nn.ReLU(True)
        )

        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(in_channels + (out_channels * len(bin_sizes)) + inter_channels2*5, out_channels,
        #               kernel_size=3, padding=1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1)
        # )

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(512 + out_channels * len(bin_sizes) + in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )


    def _make_psp_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]

        # pyramids = []
        # pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
        #                                align_corners=True) for stage in self.stages])
        # # the dense aspp block will have the original features
        # pyramids.extend([self.dense_aspp_block(features)])
        # output = self.bottleneck(torch.cat(pyramids, dim=1))

        # note for  11-02_13-28: did not have extra convolutions
        # note for 11-03_18-00: 3 step conv post head
        #   - seprable conv after each head, then bottleneck, then classifier
        # note for 11-04_09-45 -> added input featuremaps to bottleneck
        # note for 11-05_12-17 -> go back to old resnet50 (trains faster)
        # note for 11-05_17-59 -> add more conv layers to  learn deeper
        # note for 11-06 -> psp not learned extra
        # note for 11-06 2nd -> add dialated back
        # note for 11-07 -> try adam optimizer
        # psp = [features]
        # psp.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
        #                                align_corners=True) for stage in self.stages])
        # x1 = self.psp_separable(torch.cat(psp, dim=1))

        psp = [F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages]

        # the dense aspp block will have the original features
        aspp = [self.dense_aspp_block(features)]
        x2 = self.daspp_separable(torch.cat(aspp, dim=1))
        # use features here
        output = self.bottleneck1(torch.cat((features, *psp, x2), dim=1))
        return output


class PSPnet_plus(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='dialated_resnet50', inter_channels1=128, inter_channels2=64,
                 pretrained=True, use_aux=True, freeze_bn=False, freeze_backbone=False, output_stride=8):
        super(PSPnet_plus, self).__init__()
        norm_layer = nn.BatchNorm2d
        assert backbone == 'dialated_resnet50' or backbone == 'resnet50'
        self.dialated = (backbone == 'dialated_resnet50')
        self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained,
                               dialated=self.dialated)

        m_out_sz = 2048
        self.use_aux = use_aux

        self.master_branch = nn.Sequential(
            _PSPPlusModule(m_out_sz, inter_channels1=inter_channels1, inter_channels2=inter_channels2,
                           bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
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

import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
    
class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
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
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

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

class DeepPSP(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', pretrained=True, 
                 use_aux=True, dilated=True, output_stride=16, freeze_bn=False, temp_softmax=False, freeze_backbone=False,
                 hdc=False, hdc_dilation_bigger=False, **kwargs):
        super(DeepPSP, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.use_aux = use_aux
        bin_sizes = [1, 2, 3, 6]
        m_out_sz = 2048
        assert dilated != hdc
        self.backbone = ResNet(in_channels=in_channels, dilated=dilated, output_stride=output_stride, 
                               pretrained=pretrained, backbone=backbone, hdc=hdc, hdc_dilation_bigger=hdc_dilation_bigger)
        
        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz//2, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)

        if temp_softmax:
            self.T = torch.nn.Parameter(torch.Tensor([1.00]))
        else:
            self.T = 1

        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x, low_level_features, x_aux = self.backbone(x)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.backbone.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

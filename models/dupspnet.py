import math
import torch
import torch.nn.functional as F
from torch import nn
from models import resnet
from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain

class DUC(nn.Module):
    def __init__(self, in_channels, out_channels, upscale):
        super(DUC, self).__init__()
        out_channles = out_channels * (upscale ** 2)
        self.conv = nn.Conv2d(in_channels, out_channles, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU(inplace=True)
        self.pixl_shf = nn.PixelShuffle(upscale_factor=upscale)
        self.conv_w = nn.Conv2d(in_channels, out_channels * upscale * upscale, 1, bias=False)

        initialize_weights(self)
        kernel = self.icnr(self.conv.weight, scale=upscale)
        self.conv.weight.data.copy_(kernel)

    def forward(self, x):
        # x = self.relu(self.bn(self.conv(x)))
        x = self.conv_w(x)
        x = self.pixl_shf(x)
        return x

    def icnr(self, x, scale=2, init=nn.init.kaiming_normal):
        '''
        Even with pixel shuffle we still have check board artifacts,
        the solution is to initialize the d**2 feature maps with the same
        radom weights: https://arxiv.org/pdf/1707.02937.pdf
        '''
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

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
    
    
class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) 
                                                        for b_s in bin_sizes])
        low_level_channels = 1088
        inter_channels = 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(low_level_channels +(out_channels * len(bin_sizes)) + inter_channels, out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse = FeatureFused(in_channels, inter_channels=inter_channels, norm_layer=norm_layer)

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features, low_level_features):
        h, w = low_level_features.size()[2], low_level_features.size()[3]
        pyramids = [self.fuse(features, low_level_features)]
        
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        # print('pyramaids',torch.cat(pyramids, dim=1).size())
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        # print('output')
        return output

class FeatureFused(nn.Module):
    """Module for fused features"""

    def __init__(self, high_level_channels, inter_channels=512, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(high_level_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )

    def forward(self, x, low_level_features):
        size = low_level_features.size()[2:]
        x = self.conv2(F.interpolate(x, size, mode='bilinear', align_corners=True))
        fused_feature = torch.cat([x, low_level_features], dim=1)
        return fused_feature

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

class PSPDUNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', pretrained=True, 
                 use_aux=True, dilated=False, output_stride=16, freeze_bn=False, temp_softmax=False, freeze_backbone=False,
                 hdc=False, hdc_dilation_bigger=False, **kwargs):
        super(PSPDUNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.use_aux = use_aux
        bin_sizes = [1, 2, 3, 6]
        m_out_sz = 2048

        self.backbone = ResNet(in_channels=in_channels, dilated=dilated, output_stride=output_stride, 
                               pretrained=pretrained, backbone=backbone, hdc=hdc, hdc_dilation_bigger=hdc_dilation_bigger)
        self.psp_module = _PSPModule(m_out_sz, bin_sizes=bin_sizes, norm_layer=norm_layer)
        # self.dupsample = DUpsampling(m_out_sz // 4, num_classes, scale_factor=output_stride)
        self.dupsample = DUC(m_out_sz // 4, num_classes, upscale=output_stride)
        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz//2, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        if temp_softmax:
            self.T = torch.nn.Parameter(torch.Tensor([1.00]))
        else:
            self.T = 1

        initialize_weights(self.psp_module, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    # TODO - 3. lr schedulers 4. focal loss

    # TODO - training: ll feautres -> before pooling, after pooling
    # TODO - learn maps
    # TODO - mash
    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x, low_level_features, x_aux = self.backbone(x)
        x = self.psp_module(x, low_level_features)
        output = self.dupsample(x)
        output = output[:, :, :input_size[0], :input_size[1]]
        if self.training:
            output = output / self.T

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            # aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.backbone.parameters())

    def get_decoder_params(self):
        return chain(self.psp_module.parameters(), self.dupsample.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

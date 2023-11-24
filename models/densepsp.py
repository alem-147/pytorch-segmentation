import math
import torch
import torch.nn.functional as F
from torch import nn
from models import resnet
from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain

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
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d):
        super(DensePSPConv, self).__init__()
        self.add_module('aap', nn.AdaptiveAvgPool2d(output_size=bin_sz))
        self.add_module('conv', SeparableConv2d(in_channels, out_channels, 1))
        self.add_module('bn', norm_layer(out_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('dropout', nn.Dropout2d(p=drop_rate))

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        features = F.interpolate(super(DensePSPConv, self).forward(x), size=(h, w), mode='bilinear', align_corners=True)
        return features
    
class _DensePSPStage(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, bin1, bin2,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DensePSPStage, self).__init__()
        '''
        input -> binconv(bin1) -> cat(input, bin1) -> binconv(bin2) ->
        '''

        self.bin1 = DensePSPConv(in_channels, inter_channels, bin1, 
                                 drop_rate=drop_rate, norm_layer=norm_layer)
        self.bin2 = DensePSPConv(in_channels+inter_channels, out_channels, bin2,
                                 drop_rate=drop_rate, norm_layer=norm_layer)
    def forward(self, x):
        bin1 = self.bin1(x)
        x = torch.cat((x,bin1), dim=1)
        bin2 = self.bin2(x)
        return bin2
    
class _DensePSPBin(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, bins,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DensePSPBin, self).__init__()
        assert len(bins) == 2
        bin1 = bins[0]
        bin2 = bins[1]
        self.stage1 = _DensePSPStage(in_channels, inter_channels, out_channels, 
                                     bin1, bin2, drop_rate=drop_rate, norm_layer=norm_layer)
        self.stage2 = _DensePSPStage(in_channels, inter_channels, out_channels, 
                                     bin2, bin1, drop_rate=drop_rate, norm_layer=norm_layer)

    def forward(self, x):
        '''
        Stage n, n+1 -> out1
        Stage n+1, n -> out2
        out: cat(out1, out2)
        '''
        out1 = self.stage1(x)
        out2 = self.stage2(x)
        x = torch.cat((out1,out2),dim=1)

        return x

# TODO - 16, 32, 64, 128 added maps
# TODO - add dropout
# TODO - check global vs maxp for agregated feat
class _DensePSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, inter_channels, norm_layer):
        super(_DensePSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        assert bin_sizes == [1,2,3,6]

        bin1 = DensePSPConv(in_channels, out_channels, bin_sizes[0], norm_layer=norm_layer)
        bin23 = _DensePSPBin(in_channels, inter_channels, out_channels,
                                  (bin_sizes[1],bin_sizes[2]), norm_layer=norm_layer)
        bin6 = DensePSPConv(in_channels, out_channels, bin_sizes[3], norm_layer=norm_layer)
        self.stages = nn.ModuleList([bin1, bin23, bin6])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
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
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', inter_channels=64,
                  pretrained=True, use_aux=True, freeze_bn=False, freeze_backbone=False,
                 bin_sizes = [1,2,3,6]):
        super(DensePSP, self).__init__()
        norm_layer = nn.BatchNorm2d
        # by default resolve to torchvision.models.resnet152(True, norm_layer) - this loads the prtrained weights in a depricated manner
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux 

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _DensePSPModule(m_out_sz, bin_sizes, inter_channels, norm_layer=norm_layer),
            nn.Conv2d(m_out_sz//len(bin_sizes), num_classes, kernel_size=1)
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
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

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
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                   self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

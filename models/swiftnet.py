
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from models import resnet
# from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
# from libs.core.operators import dsn, upsample, conv3x3
import torch.utils.checkpoint as cp

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def dsn(in_channels, nclass, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        norm_layer(in_channels),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels, nclass, kernel_size=1, stride=1, padding=0, bias=True)
    )

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=True)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out
# 12392546 params
class SwiftNetSingleScale(BaseModel):
    def __init__(self, num_classes, backbone='resnet18', num_features=128, k_up=3, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, pretrained=True, use_aux=True, freeze_bn=False, freeze_backbone=False):
        super(SwiftNetSingleScale, self).__init__()
        self.inplanes = 64
        self.use_aux = use_aux
        self.use_bn = use_bn

        model = getattr(resnet, backbone)(pretrained, prerelres=True, norm_layer=nn.BatchNorm2d if use_bn else None)
        self.use_aux = use_aux

        # don't do deep base until you set the planes right
        self.initial = nn.Sequential(*list(model.children())[:4])

        upsamples = []
        self.layer1, self.inplanes = model.layer1, 64
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer2, self.inplanes = model.layer2, 128
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer3, self.inplanes = model.layer3, 256
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer4, self.inplanes = model.layer4, 512

        # self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        # if self.use_bn:
        #     self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.dsn = dsn(256, num_classes)
        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn)
        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        self.logits = nn.Sequential(nn.BatchNorm2d(num_features) if self.use_bn else None,
                                    nn.ReLU(inplace=self.use_bn),
                                    nn.Conv2d(num_features, num_classes, kernel_size=1))

        # self.random_init = [self.spp, self.upsample]

        self.num_features = num_classes

        # may need to modify where this does this bc resnet is pretrained
        print(self.modules())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:

            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        x = self.initial(image)
        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]

        dsn = None
        if self.training and self.use_aux:
            dsn = self.dsn(x)

        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]

        if self.training and self.use_aux:
            return features, dsn
        else:
            return features

    def forward_up(self, features):
        features = features[::-1]

        x = features[0]

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        dsn = None
        if self.training and self.use_aux:
            features, dsn = self.forward_down(x)
        else:
            features = self.forward_down(x)
        res = self.forward_up(features)
        res = self.logits(res)
        if self.training and self.use_aux:
            res.append(dsn)
            for i, out in enumerate(res):
                res[i] = upsample(out, input_size)
        else:
            res = F.interpolate(res, size=input_size, mode='bilinear')
        return res

    def get_backbone_params(self):
        return chain(self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters(),
                   self.layer4.parameters(), self.initial.parameters())
    def get_decoder_params(self):
        return chain(self.spp.parameters(),self.upsample.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class SpatialPyramidPooling(nn.Module):
    """
        SPP module is little different from ppm by inserting middle level feature to save the computation and  memory.
    """
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(_Upsample, self).__init__()
        # print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x

def convkxk(in_planes, out_planes, stride=1, k=3):
    """kxk convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride, padding=k // 2, bias=False)

def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = norm(conv(x))
        if relu is not None:
            x = relu(x)
        return x

    return bn_function
def do_efficient_fwd(block, x, efficient):
    # return block(x)
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True, k=3):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=k, batch_norm=use_bn)
        self.upsampling_method = upsample

    def forward(self, x, skip):
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x

class SWPyrBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True,
                 norm_layer=nn.BatchNorm2d, levels=3, **kwargs):
        super(SWPyrBlock, self).__init__()
        self.conv1 = convkxk(inplanes, planes, stride)
        self.bn1 = nn.ModuleList([norm_layer(planes) for _ in range(levels)])
        self.relu_inp = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convkxk(planes, planes)
        self.bn2 = nn.ModuleList([norm_layer(planes) for _ in range(levels)])
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.num_levels = levels

    def forward(self, x, level):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1[level], self.relu_inp)
        bn_2 = _bn_function_factory(self.conv2, self.bn2[level])

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        relu = self.relu(out)

        return relu, out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(SWPyrBlock, self)._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys,
                                                      unexpected_keys, error_msgs)
        missing_keys = []
        unexpected_keys = []
        for bn in self.bn1:
            bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)
        for bn in self.bn2:
            bn._load_from_state_dict(state_dict, prefix + 'bn2.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)

class SwiftNetPyramid(BaseModel):
    def __init__(self, num_classes, backbone='pyr_resnet18', num_features=128, pyramid_levels=3, use_bn=True, k_bneck=1, k_upsample=3,
                 align_corners=None, pyramid_subsample='bicubic', output_stride=4, freeze_bn=False, freeze_backbone=False,
                 pretrained=True, use_aux=False, **kwargs):
        super(SwiftNetPyramid, self).__init__()
        self.use_aux = use_aux
        self.inplanes = 64

        self.use_bn = use_bn
        bn_class = nn.BatchNorm2d if use_bn else Identity

        self.pyramid_levels = pyramid_levels
        self.num_features = num_features
        self.replicated = False

        self.align_corners = align_corners
        self.pyramid_subsample = pyramid_subsample

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.ModuleList([bn_class(64) for _ in range(pyramid_levels)])
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        model = getattr(resnet, backbone)(pretrained, prerelres=True, norm_layer=nn.BatchNorm2d if use_bn else None,
                                          pyramid_levels=pyramid_levels)

        self.initial = nn.Sequential(*list(model.children())[:4])
        self.conv1 = self.initial[0]
        self.bn1 = self.initial[1]
        self.relu = self.initial[2]
        self.maxpool = self.initial[3]

        bottlenecks = []
        # self.layer1 = self._make_layer(block, 64, layers[0], bn_class=bn_class)
        self.layer1, self.inplanes = model.layer1, 64
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_class=bn_class)
        self.layer2, self.inplanes = model.layer2, 128
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_class=bn_class)
        self.layer3, self.inplanes = model.layer3, 256
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_class=bn_class)
        self.layer4, self.inplanes = model.layer4, 512
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]


        # num_bn_remove = max(0, int(log2(output_stride) - 2))
        num_bn_remove = 0
        self.num_skip_levels = self.pyramid_levels + 3 - num_bn_remove
        bottlenecks = bottlenecks[num_bn_remove:]

        self.upsample_bottlenecks = nn.ModuleList(bottlenecks[::-1])
        num_pyr_modules = 2 + pyramid_levels - num_bn_remove

        self.upsample_blends = nn.ModuleList(
            [_UpsampleBlend(num_features,
                            use_bn=use_bn,
                            k=k_upsample)
             for i in range(num_pyr_modules)])

        self.logits = nn.Sequential(bn_class(num_features),
                                    nn.ReLU(inplace=self.use_bn),
                                    nn.Conv2d(num_features, num_classes, kernel_size=1))

        initialize_weights(self.upsample_bottlenecks, self.upsample_blends, self.logits)

    def _make_layer(self, block, planes, blocks, stride=1, bn_class=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn_class(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, levels=self.pyramid_levels)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, levels=self.pyramid_levels))

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers, idx):
        skip = None
        for l in layers:
            x = l(x) if not isinstance(l, SWPyrBlock) else l(x, idx)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image, skips, idx=-1):
        x = self.conv1(image)
        x = self.bn1[idx](x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = []
        x, skip = self.forward_resblock(x, self.layer1, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4, idx)
        features += [skip]

        skip_feats = [b(f) for b, f in zip(self.upsample_bottlenecks, reversed(features))]

        for i, s in enumerate(reversed(skip_feats)):
            skips[idx + i] += [s]

        return skips

    def forward(self, image):
        input_size = (image.size()[2], image.size()[3])
        pyramid = [image]
        for l in range(1, self.pyramid_levels):
            pyramid += [F.interpolate(image, scale_factor=1 / 2 ** l, mode=self.pyramid_subsample,
                                      align_corners=self.align_corners)]
        skips = [[] for _ in range(self.num_skip_levels)]
        additional = {'pyramid': pyramid}
        for idx, p in enumerate(pyramid):
            skips = self.forward_down(p, skips, idx=idx)
        skips = skips[::-1]
        x = skips[0][0]
        for i, (sk, blend) in enumerate(zip(skips[1:], self.upsample_blends)):
            x = blend(x, sum(sk))
        if self.use_aux == True:
            #don't do this
            return x, additional
        else:
            x = self.logits(x)
            x = F.interpolate(x, size=input_size, mode='bilinear')
            return x
    def get_backbone_params(self):
        return chain(self.conv1.parameters(), self.maxpool.parameters(), self.layer1.parameters(),
                     self.layer2.parameters(), self.layer3.parameters(), self.layer4.parameters(), self.bn1.parameters())
    def get_decoder_params(self):
        return chain(self.upsample_bottlenecks.parameters(), self.upsample_blends.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     super(SwiftNetPyramid, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
    #                                               unexpected_keys, error_msgs)
    #     for bn in self.bn1:
    #         bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
    #                                  error_msgs)


if __name__ == '__main__':
    i = torch.Tensor(1, 3, 512, 512).cuda()
    m = SwiftNetSingleScale(pretrained=False).cuda()
    m.eval()
    o = m(i)
    print(o[0].size())
    print("output length: ", len(o))
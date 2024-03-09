import os

import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from skimage.feature import canny
from collections import OrderedDict


def actLayer(kind='relu'):
    if kind == 'tanh':
        return nn.Tanh()
    elif kind == 'sigmoid':
        return nn.Sigmoid()
    elif kind == 'relu':
        return nn.ReLU(inplace=True)
    elif kind == 'leaky':
        return nn.LeakyReLU(0.2, inplace=True)
    elif kind == 'elu':
        return nn.ELU(1.0, inplace=True)
    else:
        return nn.Identity()


def normLayer(channels, kind='bn', affine=True):
    if kind == 'bn':
        return nn.BatchNorm2d(channels, affine=affine)
    elif kind == 'in':
        return nn.InstanceNorm2d(channels, affine=affine)
    else:
        return nn.Identity(channels)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                try:
                    nn.init.constant_(m.weight, 1)
                    nn.init.normal_(m.bias, 0.0001)
                except:
                    pass

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def print_networks(self, model_name):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()

        print('[Network %s] Total number of parameters : %.2f M' % (model_name, num_params / 1e6))
        print('-----------------------------------------------')


class SeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='reflect'):
        super(SeperableConv, self).__init__()
        self.depthConv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias, padding_mode=padding_mode)
        self.pointConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=bias,
                                   padding_mode=padding_mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)

        return x


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='reflect', kind='depthConv', norm_layer='',
                 activation=''):
        super(MyConv2d, self).__init__()
        if kind == 'depthConv':
            self.conv = SeperableConv(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, groups, bias, padding_mode=padding_mode)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation, groups, bias, padding_mode=padding_mode)

        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x


class MyDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='reflect', kind='depthConv', scale_mode='bilinear',
                 norm_layer='', activation=''):
        super(MyDeConv2d, self).__init__()
        if kind == 'depthConv':
            self.conv = SeperableConv(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)

        self.scale_factor = stride
        self.scale_mode = scale_mode

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor, mode=self.scale_mode)
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x


# FU in paper
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, use_spectral=False, norm_layer='in', activation='relu'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        # kernel size was fixed to 1
        # because the global receptive field.
        if not use_spectral:
            self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                              kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        else:
            self.conv_layer = nn.utils.spectral_norm(
                torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
        self.norm = normLayer(kind=norm_layer, channels=out_channels * 2)
        self.act = actLayer(kind=activation)

        nn.init.kaiming_normal_(self.conv_layer.weight)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # The FFT of a real signal is Hermitian-symmetric, X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])
        # so the full fftn() output contains redundant information.
        # rfftn() instead omits the negative frequencies in the last dimension.

        # (batch, c, h, w/2+1) complex number
        ffted = torch.fft.rfftn(x, s=(h, w), dim=(2, 3), norm='ortho')  # norm='ortho' making the real FFT orthonormal
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))

        ffted = torch.tensor_split(ffted, 2, dim=1)
        ffted = torch.complex(ffted[0], ffted[1])
        output = torch.fft.irfftn(ffted, s=(h, w), dim=(2, 3), norm='ortho')

        return output


class LFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='reflect',
                 nc_factor=4, groups_num=4, norm_layer='bn', activation='relu', use_in_act=True, use_in_norm=True,
                 use_out_act=True, use_out_norm=True, fusion='conv'):
        super(LFFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        bottleneck_nc = out_channels // nc_factor

        self.g_conv_1x1_compress = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_nc, kernel_size=1, groups=groups_num),
            normLayer(bottleneck_nc, kind=norm_layer if use_in_norm else ''),
            actLayer(kind=activation if use_in_act else ''),
        )

        self.groups_num = groups_num

        self.global_conv = FourierUnit(bottleneck_nc, bottleneck_nc, norm_layer=norm_layer, activation=activation)

        self.local_conv = SeperableConv(bottleneck_nc, bottleneck_nc, kernel_size,
                                        stride, padding, dilation, groups, bias, padding_mode=padding_mode)

        self.use_out_act = use_out_act
        if use_out_act:
            self.out_act = actLayer(kind=activation)

        self.use_out_norm = use_out_norm
        if use_out_norm:
            self.out_norm = normLayer(out_channels)

        self.fusion = fusion

        if 'double_gate' in self.fusion:

            # as a complement
            self.local_norm_act = nn.Sequential(
                normLayer(bottleneck_nc, kind=norm_layer),
                actLayer(kind=activation),
            )
            self.local_gate = nn.Sequential(
                SeperableConv(bottleneck_nc, bottleneck_nc, kernel_size,
                              stride, padding, dilation, groups, bias,
                              padding_mode=padding_mode),
                nn.Sigmoid()
            )

            self.global_gate = nn.Sequential(
                SeperableConv(bottleneck_nc, bottleneck_nc, kernel_size,
                              stride, padding, dilation, groups, bias,
                              padding_mode=padding_mode),
                nn.Sigmoid()
            )

            self.fusion_module = nn.Sequential(
                SeperableConv(2 * bottleneck_nc, bottleneck_nc, kernel_size, stride, padding, dilation,
                              groups, bias,
                              padding_mode=padding_mode),
                normLayer(bottleneck_nc, kind=norm_layer),
                actLayer(kind=activation),
            )

        elif 'gate' in self.fusion:
            self.gate = nn.Sequential(
                SeperableConv(bottleneck_nc, bottleneck_nc, kernel_size,
                              stride, padding, dilation, groups, bias,
                              padding_mode=padding_mode),
                nn.Sigmoid()
            )
            self.fusion_module = nn.Sequential(
                SeperableConv(2 * bottleneck_nc, bottleneck_nc, kernel_size, stride, padding, dilation,
                              groups, bias,
                              padding_mode=padding_mode),
                normLayer(bottleneck_nc, kind=norm_layer),
                actLayer(kind=activation),
            )



        elif 'conv' in self.fusion:
            self.fusion_module = nn.Sequential(
                SeperableConv(2 * bottleneck_nc, bottleneck_nc, kernel_size, stride, padding, dilation,
                              groups, bias,
                              padding_mode=padding_mode),
            )

        if fusion == 'gate_conv_concat':
            merged_nc = 2 * bottleneck_nc
            merged_groups = 2 * groups_num
        elif fusion == 'concat':
            merged_nc = 2 * bottleneck_nc
            merged_groups = groups_num
        else:
            merged_nc = bottleneck_nc
            merged_groups = groups_num

        self.g_conv_1x1_expand = nn.Sequential(
            nn.Conv2d(merged_nc, out_channels, kernel_size=1, groups=merged_groups),
            normLayer(out_channels, kind=norm_layer if use_out_norm else ''),
            actLayer(kind=activation if use_out_act else ''),
        )

    def forward(self, input):
        x = self.g_conv_1x1_compress(input)
        x_g = self.global_conv(x)
        x_l = self.local_conv(x)

        if self.fusion == 'gate_conv_concat':
            gamma = self.gate(x)
            merge_x = torch.cat((x_l, x_g), dim=1)
            merge_x = self.fusion_module(merge_x)
            merge_x = merge_x * gamma
            merge_x = torch.cat((merge_x, x), dim=1)
        elif self.fusion == 'gate_conv_residual':
            gamma = self.gate(x)
            merge_x = torch.cat((x_l, x_g), dim=1)
            merge_x = self.fusion_module(merge_x)
            merge_x = merge_x * gamma
            merge_x = merge_x + x

        # use gated func before
        elif self.fusion == 'double_gate':
            gamma_l = self.local_gate(x)
            x_l = self.local_norm_act(x_l)
            x_l = gamma_l * x_l

            gamma_g = self.global_gate(x)
            x_g = gamma_g * x_g

            merge_x = torch.cat((x_l, x_g), dim=1)
            merge_x = self.fusion_module(merge_x)

        # use gated func after
        elif self.fusion == 'double_gate_v2':
            x_l = self.local_norm_act(x_l)
            gamma_l = self.local_gate(x_l)
            x_l = gamma_l * x_l

            gamma_g = self.global_gate(x_g)
            x_g = gamma_g * x_g

            merge_x = torch.cat((x_l, x_g), dim=1)
            merge_x = self.fusion_module(merge_x)

        elif self.fusion == 'conv_residual':
            merge_x = torch.cat((x_l, x_g), dim=1)
            merge_x = self.fusion_module(merge_x)
            merge_x = merge_x + x

        elif self.fusion == 'conv':
            # simply fusion
            merge_x = torch.cat((x_l, x_g), dim=1)
            merge_x = self.fusion_module(merge_x)
        else:
            # simply concat
            merge_x = torch.cat((x_l, x_g), dim=1)

        out_x = self.g_conv_1x1_expand(merge_x)

        return out_x


class GatedConv2dWithAct(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_layer='bn', activation='relu', mask_conv='depthConv', conv='normal', padding_mode='reflect'):
        super(GatedConv2dWithAct, self).__init__()
        self.conv2d = MyConv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode=padding_mode, kind=conv)
        self.mask_conv2d = MyConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation, groups, bias, padding_mode=padding_mode, kind=mask_conv)
        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)
        self.gated = actLayer(kind='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = self.act(self.norm(x))
        x = x * self.gated(mask)

        return x


class GatedLFFConvWithAct(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, norm_layer='bn', activation='relu', mask_conv='depthConv',
                 nc_factor=2, groups_num=1, padding_mode='reflect', fusion='conv',
                 use_in_norm=False, use_in_act=False, use_out_norm=False, use_out_act=False,enable_lfu=False):
        super(GatedLFFConvWithAct, self).__init__()
        self.conv2d = LFFC(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                           nc_factor=nc_factor, groups_num=groups_num, norm_layer=norm_layer, activation=activation,
                           fusion=fusion,
                           use_in_norm=use_in_norm, use_in_act=use_in_act, use_out_norm=use_out_norm,
                           use_out_act=use_out_act)
        self.mask_conv2d = MyConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation, groups, bias, padding_mode=padding_mode, kind=mask_conv)
        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)
        self.gated = actLayer(kind='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = self.act(self.norm(x))
        x = x * self.gated(mask)

        return x


class GatedDeConv2d(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, norm_layer='in', activation='leaky', mask_conv='depth', conv='normal'):
        super(GatedDeConv2d, self).__init__()
        self.conv2d = GatedConv2dWithAct(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                         dilation=dilation, groups=groups, bias=bias, norm_layer=norm_layer,
                                         activation=activation, mask_conv=mask_conv, conv=conv)
        self.scale_factor = stride

    def forward(self, input):
        # print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class GatedLFFDeConv(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, norm_layer='bn', activation='relu', mask_conv='depthConv',
                 nc_factor=2, groups_num=1, padding_mode='reflect', fusion='conv',
                 use_in_norm=False, use_in_act=False, use_out_norm=False, use_out_act=False):
        super(GatedLFFDeConv, self).__init__()
        self.conv2d = GatedLFFConvWithAct(in_channels, out_channels,
                                          kernel_size, stride=1, padding=padding, dilation=dilation,
                                          groups=groups, bias=bias, norm_layer=norm_layer,
                                          activation=activation, mask_conv=mask_conv, nc_factor=nc_factor,
                                          groups_num=groups_num,
                                          padding_mode=padding_mode, fusion=fusion,
                                          use_in_norm=use_in_norm, use_in_act=use_in_act,
                                          use_out_norm=use_out_norm, use_out_act=use_out_act)
        self.scale_factor = stride

    def forward(self, input):
        # print(input.size())
        x = F.interpolate(input, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv2d(x)


class LFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_mode, norm_layer='bn', activation='relu',
                 dilation=1, nc_factor=4, groups_num=4, fusion='conv'):
        super().__init__()
        self.conv1 = LFFC(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                          norm_layer=norm_layer,
                          activation=activation,
                          padding_mode=padding_mode,
                          nc_factor=nc_factor,
                          groups_num=groups_num,
                          fusion=fusion,
                          use_out_norm=True,
                          use_out_act=True,
                          )
        self.conv2 = LFFC(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                          norm_layer=norm_layer,
                          activation=activation,
                          padding_mode=padding_mode,
                          nc_factor=nc_factor,
                          groups_num=groups_num,
                          fusion=fusion,
                          use_out_norm=True,
                          use_out_act=False,
                          )

    def forward(self, x):
        residual = x
        out_x = self.conv1(x)
        out_x = self.conv2(out_x)
        out_x = out_x + residual
        return out_x


# sequence flow
class MyLFFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 scale_factor=2, norm_layer='in', activation='relu', padding_mode='reflect'):
        super(MyLFFC, self).__init__()
        # reduce channel
        self.conv = SeperableConv(in_channels, out_channels // scale_factor, kernel_size, 1, padding, dilation, groups,
                                  bias, padding_mode=padding_mode)

        self.fu = FourierUnit(out_channels // scale_factor, out_channels // scale_factor, norm_layer=norm_layer,
                              activation=activation)

        # merge conv
        self.merge_conv = nn.Conv2d(in_channels=(out_channels * 2 // scale_factor), out_channels=out_channels,
                                    kernel_size=1, stride=stride,
                                    padding=0, bias=False)

        # initial weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        fu_x = self.fu(x)
        cat_x = torch.cat((x, fu_x), dim=1)
        out = self.merge_conv(cat_x)

        return out

class LSPADE(nn.Module):
    def __init__(self, norm_nc, label_nc,norm_layer='bn'):
        super().__init__()

        self.param_free_norm = normLayer(norm_nc,kind=norm_layer,affine=False)
        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        # ks = int(parsed.group(2))
        #
        # if param_free_norm_type == 'instance':
        #     self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'batch':
        #     self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # else:
        #     raise ValueError('%s is not a recognized param-free norm type in SPADE'
        #                      % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        # pw = ks // 2
        self.mlp_shared = nn.Sequential(
            SeperableConv(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = SeperableConv(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = SeperableConv(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class LSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc,norm_layer='bn'):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = SeperableConv(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = SeperableConv(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = SeperableConv(fin, fout, kernel_size=1, bias=False)

        # # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # # define normalization layers
        # spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = LSPADE(fin, semantic_nc,norm_layer)
        self.norm_1 = LSPADE(fmiddle, semantic_nc,norm_layer)
        if self.learned_shortcut:
            self.norm_s = LSPADE(fin, semantic_nc,norm_layer)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1,inplace=True)

class GatedLFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_mode = 'reflect', norm_layer='bn', activation='relu',
                 dilation=1,nc_factor=4,groups_num = 4,fusion ='conv',enable_lfu=False):
        super().__init__()
        self.conv1 = GatedLFFConvWithAct(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation=activation,
                                padding_mode=padding_mode,
                                nc_factor = nc_factor,
                                groups_num = groups_num,
                                fusion=fusion,
                                enable_lfu=enable_lfu,
                                use_out_norm=False,
                                use_out_act=False,
                                )
        self.conv2 = GatedLFFConvWithAct(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation=activation,
                                padding_mode=padding_mode,
                                nc_factor=nc_factor,
                                groups_num=groups_num,
                                fusion=fusion,
                                enable_lfu=enable_lfu,
                                use_out_norm=False,
                                use_out_act=False,
                                )

        # self.res_gate  = nn.Sequential(
        #     SeperableConv(dim,dim,kernel_size=3,padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        residual = x
        out_x = self.conv1(x)
        out_x = self.conv2(out_x)
        # gamma = self.res_gate(residual)
        # out_x = out_x * gamma + residual * (1 - gamma)  #focus on the missing area, keep the background unchanged
        out_x = out_x + residual
        return out_x

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1,groups=1, enable_lfu=True,norm_layer='bn',activation='relu'):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.enable_lfu = enable_lfu
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            normLayer(out_channels //2,kind=norm_layer),
            actLayer(kind=activation)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups,norm_layer=norm_layer,activation=activation)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups,norm_layer=norm_layer,activation=activation)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

#light-weight version of original FFC
class NoFusionLFFC(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1,dilation=1, groups=1, bias=True,padding_mode='reflect',
                 norm_layer='bn', activation='relu',enable_lfu=False,ratio_g_in=0.5,ratio_g_out=0.5,nc_reduce=2,
                 out_act=True):
        super(NoFusionLFFC, self).__init__()
        self.ratio_g_in = ratio_g_in
        self.ratio_g_out = ratio_g_out
        in_cg = int(in_channels * ratio_g_in)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_g_out)
        out_cl = out_channels - out_cg

        if in_cl >0 and nc_reduce > 1:
            self.l_in_conv = nn.Sequential(
                nn.Conv2d(in_cl, in_cl // nc_reduce, kernel_size=1),
                normLayer(channels=in_cl // nc_reduce, kind=norm_layer),
                actLayer(kind=activation)
            )
        else:
            self.l_in_conv = nn.Identity()

        if out_cl >0 and nc_reduce >1:
            self.out_L_bn_act = nn.Sequential(
                nn.Conv2d(out_cl // nc_reduce, out_cl, kernel_size=1),
                normLayer(channels=out_cl, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        elif out_cl >0:
            self.out_L_bn_act = nn.Sequential(
                normLayer(channels=out_cl, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        else:
            self.out_L_bn_act = nn.Identity()

        if in_cg >0 and nc_reduce > 1:
            self.g_in_conv =  self.g_in_conv = nn.Sequential(
                nn.Conv2d(in_cg, in_cg // nc_reduce, kernel_size=1),
                normLayer(channels=in_cg // nc_reduce, kind=norm_layer),
                actLayer(kind=activation)
            )
        else:
            self.g_in_conv = nn.Identity()

        if out_cg >0 and nc_reduce > 1:
            self.out_G_bn_act = nn.Sequential(
                nn.Conv2d(out_cg // nc_reduce, out_cg, kernel_size=1),
                normLayer(channels=out_cg, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        elif out_cg >0:
            self.out_G_bn_act = nn.Sequential(
                normLayer(channels=out_cg, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        else:
            self.out_G_bn_act = nn.Identity()

        module = nn.Identity if in_cl == 0 or out_cl == 0 else SeperableConv
        self.convl2l = module(in_cl // nc_reduce, out_cl // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else SeperableConv
        self.convl2g = module(in_cl // nc_reduce, out_cg // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else SeperableConv
        self.convg2l = module(in_cg // nc_reduce, out_cl // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform

        self.convg2g = module(in_cg // nc_reduce, out_cg // nc_reduce, stride=stride,
            norm_layer=norm_layer,activation=activation, enable_lfu=enable_lfu)

        self.feats_dict = {}
        self.flops = 0

    def flops_count(self,module,input):
        if isinstance(module,nn.Module) and not isinstance(module,nn.Identity):
            if isinstance(input,torch.Tensor):
                # input_shape = input.shape[1:]
                flops = flop_counter(module,input)
                if flops != None:
                    self.flops += flops

    def get_flops(self):
        for m_name,input in self.feats_dict.items():
            module = getattr(self,m_name)
            self.flops_count(module,input)

        print(f'Total FLOPs : {self.flops:.5f} G')

        return self.flops

    def forward(self,x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        # self.feats_dict['l_in_conv'] = x_l
        # self.feats_dict['g_in_conv'] = x_g
        x_l,x_g = self.l_in_conv(x_l),self.g_in_conv(x_g)


        if self.ratio_g_out != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
            # self.feats_dict['convl2l'] = x_l
            # self.feats_dict['convg2l'] = x_g
            # self.feats_dict['out_L_bn_act'] = out_xl
            out_xl = self.out_L_bn_act(out_xl)
        if self.ratio_g_out != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
            # self.feats_dict['convl2l'] = x_l
            # self.feats_dict['convg2l'] = x_g
            # self.feats_dict['out_G_bn_act'] = out_xg
            out_xg = self.out_G_bn_act(out_xg)

        return out_xl, out_xg

class TwoStreamLFFCResNetBlock(nn.Module):
    def __init__(self,dim,kernel_size=3,padding=1,norm_layer='bn',activation='relu',nc_reduce=2,
                 ratio_g_in=0.5,ratio_g_out=0.5):
        super(TwoStreamLFFCResNetBlock, self).__init__()
        self.fusion = 0
        self.conv1 = NoFusionLFFC(in_channels=dim,out_channels=dim,kernel_size=kernel_size,padding=padding,
                                  norm_layer=norm_layer,activation=activation,out_act=True,nc_reduce=nc_reduce,
                                  ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out)
        self.conv2 = NoFusionLFFC(in_channels=dim, out_channels=dim, kernel_size=kernel_size,padding=padding,
                                  norm_layer=norm_layer, activation=activation, out_act=False, nc_reduce=nc_reduce,
                                  ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out)

    def get_flops(self):
        self.conv1.get_flops()
        self.conv2.get_flops()
        self.flops += self.conv1.flops + self.conv2.flops
        print(f'Total FLOPs : {self.flops:.5f} G')
        return self.flops


    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = self.conv1(x)
        x_l, x_g = self.conv2(x)

        out_x_l = x_l + id_l
        out_x_g = x_g + id_g
        return out_x_l, out_x_g

# use NoFusion LFFC as basic block
# inherently seperate local and global branch
# use the global output xg to learn holistic structure
# then inject it by SPADE into local detail generation
# nc_reduce = 2
# replace BN in SPAED with IN
# without inference-free co-learning branch
class DistInpaintModel_SPADE_IN_LFFC_Base_wo_ifcb(BaseNetwork):
    def __init__(self, input_nc = 5, output_nc = 3, ngf=64, n_downsampling=3, n_blocks=9, merge_blocks=3,
                 norm_layer='bn', padding_mode='reflect', activation='relu',
                 out_act='tanh', max_features=512, nc_reduce=2, ratio_g_in=0.5, ratio_g_out=0.5,
                 selected_edge_layers=[], selected_gt_layers=[], enable_lfu=False, is_training=True):
        assert (n_blocks >= 0)
        super().__init__()
        self.is_training = is_training
        self.num_down = n_downsampling
        self.merge_blks = merge_blocks
        self.n_blocks = n_blocks
        self.selected_edge_layers = selected_edge_layers
        self.selected_gt_layers = selected_gt_layers
        self.en_l0 = nn.Sequential(nn.ReflectionPad2d(2),
                                   NoFusionLFFC(in_channels=input_nc, out_channels=ngf, kernel_size=5, padding=0,
                                                ratio_g_in=0, ratio_g_out=0, nc_reduce=1, norm_layer=norm_layer,
                                                activation=activation))

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == 0:
                g_in = 0
            else:
                g_in = ratio_g_in
            model = NoFusionLFFC(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 ratio_g_in=g_in, ratio_g_out=ratio_g_out,
                                 nc_reduce=1, norm_layer=norm_layer, activation=activation)
            setattr(self, f"en_l{i + 1}", model)

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TwoStreamLFFCResNetBlock(feats_num_bottleneck, nc_reduce=nc_reduce, ratio_g_in=ratio_g_in,
                                                    ratio_g_out=ratio_g_out, norm_layer=norm_layer,
                                                    activation=activation)
            setattr(self, f"en_res_l{i}", cur_resblock)

        for i in range(merge_blocks):
            mult = 2 ** (n_downsampling - 1)
            model = LSPADEResnetBlock(fin=ngf * mult, fout=ngf * mult, semantic_nc=ngf * mult,
                                      norm_layer='in')

            setattr(self, f"merge_l{i}", model)

        ### texture decoder
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i - 1)
            model = GatedDeConv2d(min(max_features, ngf * mult),
                                  min(max_features, int(ngf * mult / 2)),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer, activation=activation
                                  )

            setattr(self, f'de_l{i}', model)

        out_model = [nn.ReflectionPad2d(2),
                     GatedConv2dWithAct(ngf // 2, output_nc, kernel_size=5, stride=1, padding=0,
                                        norm_layer='', activation='', mask_conv='depthConv', conv='normal')]
        out_model.append(actLayer(kind=out_act))
        self.out_l = nn.Sequential(*out_model)

        self.init_weights()

    def set_finetune_mode(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.InstanceNorm2d):
                module.eval()
            elif isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, x):
        selected_edge_feats = {}
        selected_gt_feats = {}
        feats_dict = {}
        if self.is_training:
            for i in range(self.num_down + 1):
                x = getattr(self, f'en_l{i}')(x)
                x_l, x_g = x
                feats_dict[f'en_l{i}_xl'] = x_l
                feats_dict[f'en_l{i}_xg'] = x_g

            for i in range(self.n_blocks):
                x = getattr(self, f'en_res_l{i}')(x)
                x_l, x_g = x
                feats_dict[f'en_res_l{i}_xl'] = x_l
                feats_dict[f'en_res_l{i}_xg'] = x_g

            x_l, x_g = x
            edge = x_g
            edge2spade = x_g
            feats_dict['edge2spade'] = x_g

            merged_feat, edge_feat = x_l, edge2spade
            for i in range(self.merge_blks):
                merged_feat = getattr(self, f'merge_l{i}')(merged_feat, edge_feat)
                feats_dict[f'merge_l{i}'] = merged_feat

            x = merged_feat
            for i in range(self.num_down):
                x = getattr(self, f'de_l{i}')(x)
                feats_dict[f'de_l{i}'] = x

            selected_gt_feats = {k: feats_dict.get(k) for k in self.selected_gt_layers}
            feats_dict.clear()
            out_edge = None

        else:
            out_edge = None

            for i in range(self.num_down + 1):
                x = getattr(self, f'en_l{i}')(x)
                feats_dict[f'en_l{i}'] = x

            for i in range(self.n_blocks):
                x = getattr(self, f'en_res_l{i}')(x)

            x_l, x_g = x
            edge2spade = x_g
            feats_dict['edge2spade'] = x_g
            merged_feat, edge_feat = x_l, edge2spade
            for i in range(self.merge_blks):
                merged_feat = getattr(self, f'merge_l{i}')(merged_feat, edge_feat)

            x = merged_feat
            for i in range(self.num_down):
                x = getattr(self, f'de_l{i}')(x)

            feats_dict.clear()

        out_x = self.out_l(x)
        out_x = (out_x + 1) / 2.0
        return out_x, out_edge, selected_edge_feats, selected_gt_feats








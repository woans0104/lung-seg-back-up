import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

#########################################################################################################


# coordconv =========================================================================================

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        # print('before coordconv ',x.size())
        ret = self.addcoords(x)
        # print('after coordconv ', ret.size())
        ret = self.conv(ret)
        return ret


class CoordConv_block(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('before coordconv ',x.size())
        ret = self.addcoords(x)
        # print('after coordconv ', ret.size())
        ret = self.conv(ret)
        ret = self.sigmoid(ret)

        batch_size, channel, a, b = x.size()
        # spatial excitation
        output_tensor = torch.mul(x, ret.view(batch_size, 1, a, b))

        return output_tensor


def TF_coordconv(encodernumber, coordconv):
    TF_coordconv_list = []
    if coordconv == None:
        TF_coordconv_list = [False for i in range(encodernumber)]
    else:
        for i in range(0, encodernumber):
            if i in coordconv:
                TF_coordconv_list.append(True)
            else:
                TF_coordconv_list.append(False)

    assert len(TF_coordconv_list) == encodernumber, 'not match coordconv_list'

    return TF_coordconv_list


# coordconv =========================================================================================


# 2d model

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, coordconv=False,
                 radius=False):
        super(ConvBnRelu, self).__init__()

        if coordconv:
            # 1x1 conv
            self.conv = CoordConv(in_channels, out_channels, kernel_size=1,
                                  padding=0, stride=1, with_r=radius)
            # self.conv = CoordConv(in_channels, out_channels, kernel_size=kernel_size,
            #                     padding=1, stride=1, with_r=radius)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, coordconv=False,
                 radius=False):
        super(ConvRelu, self).__init__()

        if coordconv:
            # 1x1 conv
            self.conv = CoordConv(in_channels, out_channels, kernel_size=1,
                                  padding=0, stride=1, with_r=radius)
            # self.conv = CoordConv(in_channels, out_channels, kernel_size=kernel_size,
            #                     padding=1, stride=1, with_r=radius)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, stride=stride)

        #self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        return x



class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)
        x = self.transpose_conv(x)
        #if down_tensor != None:
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


class StackEncoder_skip(nn.Module):
    def __init__(self, in_channels, out_channels, padding, calculate, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder_skip, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # if cal = concat
        self.down_useconcat = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1)
        self.down_Bn = nn.BatchNorm2d(out_channels)
        self.calculate = calculate

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        identity = self.down(x)
        identity = self.down_Bn(identity)

        x = self.convr1(x)
        x = self.convr2(x)

        x = calculate_mode(self.calculate, x, identity)
        if self.calculate == 'concat':
            x = self.down_useconcat(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder_skip(nn.Module):
    def __init__(self, in_channels, out_channels, padding, calculate, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder_skip, self).__init__()

        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        # if cal == concat
        self.down_useconcat = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1)
        self.down_Bn = nn.BatchNorm2d(out_channels)
        self.calculate = calculate

        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)

        x = self.transpose_conv(x)
        identity = self.down(down_tensor)
        identity = self.down_Bn(identity)

        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)

        x = calculate_mode(self.calculate, x, identity)
        if self.calculate == 'concat':
            x = self.down_useconcat(x)
        return x


# use attnUnet
class StackDecoder_attn(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder_attn, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.attn_block = Attention_block(F_g=out_channels, F_l=out_channels, F_int=out_channels)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)
        x = self.transpose_conv(x)

        attn = self.attn_block(x, down_tensor)
        print('here')
        x = self._crop_concat(x, attn)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


class res_calculate():
    def sum(x, identity):
        x = x + identity
        return x

    def multiple(x, identity):
        x = x * identity
        return x

    def concat(x, identity):
        x = torch.cat((x, identity), 1)
        return x


def calculate_mode(mode, x, identy):
    if mode == 'sum':
        calculate = res_calculate.sum(x, identy)
    elif mode == 'multiple':
        calculate = res_calculate.multiple(x, identy)
    elif mode == 'concat':
        calculate = res_calculate.concat(x, identy)

    return calculate


# attention model

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        print('#' * 50)
        print(g.shape)
        g1 = self.W_g(g)
        print('@' * 50)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# Squeeze and Excitation Module


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels_reduced)
        self.relu = nn.ReLU()

        # self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        #self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.fc2(fc_out_1)
        #fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        #output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        output_tensor = fc_out_2
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        #squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        #output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        output_tensor = out

        return output_tensor


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, coordconv=True, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
        # self.coord = CoordConv_block(num_channels,num_channels)

        self.coord_mode = coordconv

        # def __init__(self, in_channels, out_channels, with_r=False, **kwargs):

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        """
        if self.coord_mode :
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor) + self.coord(input_tensor)
        else:
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)
        """

        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)

        return output_tensor


class ChannelSpatialSELayer_coord(nn.Module):

    def __init__(self, num_channels, coordconv=True, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
        self.coord = CoordConv_block(num_channels, num_channels)

        self.coord_mode = coordconv

        # def __init__(self, in_channels, out_channels, with_r=False, **kwargs):

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))

        if self.coord_mode:
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor) + self.coord(input_tensor)
        else:
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)

        return output_tensor


class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g



class StackEncoder_srm(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder_srm, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.SRMLayer = SRMLayer(out_channels)




    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        x= self.SRMLayer(x)

        return x, x_trace


#tanh

class ConvBntanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, coordconv=False,
                 radius=False):
        super(ConvBntanh, self).__init__()

        if coordconv:
            # 1x1 conv
            self.conv = CoordConv(in_channels, out_channels, kernel_size=1,
                                  padding=0, stride=1, with_r=radius)
            # self.conv = CoordConv(in_channels, out_channels, kernel_size=kernel_size,
            #                     padding=1, stride=1, with_r=radius)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.tanh(x)
        return x

class StackEncoder_tanh(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder_tanh, self).__init__()
        self.convr1 = ConvBntanh(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBntanh(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder_tanh(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder_tanh, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')

        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.convr1 = ConvBntanh(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBntanh(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)
        x = self.transpose_conv(x)
        #if down_tensor != None:
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


######################################################################################################
# model


class Unet2D(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=32):
        super(Unet2D, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel


        self.down1 = StackEncoder(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)
        self.down4 = StackEncoder(self.start_channel*4, self.start_channel*8, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*8, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(self.start_channel*16, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=self.start_channel*16, out_channels=self.start_channel*8, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        center = self.center(x)

        x = self.up1(center, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)

        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        if out.shape[-1] != self.width:
            out = self.output_up_seg_map(out)

        return out,center



class Unet2D_srm(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=32):
        super(Unet2D_srm, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel


        self.down1 = StackEncoder_srm(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder_srm(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder_srm(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)
        self.down4 = StackEncoder_srm(self.start_channel*4, self.start_channel*8, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*8, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(self.start_channel*16, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=self.start_channel*16, out_channels=self.start_channel*8, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        center = self.center(x)

        x = self.up1(center, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)

        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        if out.shape[-1] != self.width:
            out = self.output_up_seg_map(out)

        return out,center




class Unetcoordconv(nn.Module):
    """
    coordconv unet : padding=1,momentum=0.1,coordconv = True
    """

    def __init__(self, in_shape, start_channel=32, momentum=0.1, coordnumber=None, radius=False):
        super(Unetcoordconv, self).__init__()

        self.start_channel = start_channel
        channels, heights, width = in_shape
        encodernumber = 10
        padding = 1


        if coordnumber:
            TF_coordconv_list = TF_coordconv(encodernumber, coordnumber)
        else:
            TF_coordconv_list = TF_coordconv(encodernumber, coordnumber)

        self.down1 = StackEncoder(channels, self.start_channel, padding, momentum=momentum, coordconv=TF_coordconv_list[0],
                                  radius=radius)
        self.down2 = StackEncoder(self.start_channel, self.start_channel*2, padding, momentum=momentum, coordconv=TF_coordconv_list[1], radius=radius)
        self.down3 = StackEncoder(self.start_channel*2, self.start_channel*4, padding, momentum=momentum, coordconv=TF_coordconv_list[2], radius=radius)
        self.down4 = StackEncoder(self.start_channel*4, self.start_channel*8, padding, momentum=momentum, coordconv=TF_coordconv_list[3], radius=radius)

        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*8, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=TF_coordconv_list[4], radius=radius),
            ConvBnRelu(self.start_channel*16, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=TF_coordconv_list[5], radius=radius)
        )

        self.up1 = StackDecoder(in_channels=self.start_channel*16, out_channels=self.start_channel*8, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[6], radius=radius)
        self.up2 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[7], radius=radius)
        self.up3 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[8], radius=radius)
        self.up4 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[9], radius=radius)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):

        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        return out



class ae_lung(nn.Module):
    # down 4
    def __init__(self,in_shape,start_channel=32,kenel_size=3,padding=1):
        super(ae_lung, self).__init__()

        self.channels, self.heights, self.width = in_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(1, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 2, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 4, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            #center
            nn.Conv2d(start_channel * 8, start_channel * 16, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 16),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(start_channel*16, start_channel*16, 2, stride=2),
            nn.Conv2d(start_channel * 16, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*8),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 8, start_channel * 8, 2, stride=2),
            nn.Conv2d(start_channel * 8, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 4, start_channel*4, 2, stride=2),
            nn.Conv2d(start_channel * 4, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel*2 , start_channel*2, 2, stride=2),
            nn.Conv2d(start_channel * 2, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),



            nn.Conv2d(start_channel, 1, 1)

        )


    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder, encoder


class ae_lung_adversarial(nn.Module):
    # down 4
    def __init__(self,in_shape,start_channel=32,kenel_size=3,padding=1,advers_mode=True):
        super(ae_lung_adversarial, self).__init__()

        self.channels, self.heights, self.width = in_shape
        self.advers_mode = advers_mode
        self.encoder = nn.Sequential(
            nn.Conv2d(1, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 2, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 4, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            #center
            nn.Conv2d(start_channel * 8, start_channel * 16, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 16),
            nn.ReLU(),

        )


        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(start_channel*16, start_channel*16, 2, stride=2),
            nn.Conv2d(start_channel * 16, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*8),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 8, start_channel * 8, 2, stride=2),
            nn.Conv2d(start_channel * 8, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 4, start_channel*4, 2, stride=2),
            nn.Conv2d(start_channel * 4, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel*2 , start_channel*2, 2, stride=2),
            nn.Conv2d(start_channel * 2, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),



            nn.Conv2d(start_channel, 1, 1)

        )

    def _random_sampling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()


        random_size, _, _ = channel_mean.shape
        bottom_ae_randsample_ = np.random.normal(np.array(channel_mean.cpu().data),
                                                 np.array(channel_std.cpu().data),
                                                 size=(random_size, 512, 16 * 16))
        bottom_ae_randsample = torch.from_numpy(bottom_ae_randsample_.reshape(random_size, 512, 16, 16))
        bottom_ae_randsample = bottom_ae_randsample.type("torch.FloatTensor").cuda()

        return bottom_ae_randsample


    def forward(self, x):
        encoder = self.encoder(x)
        random = self._random_sampling(encoder)
        decoder = self.decoder(random)



        return decoder, random



class Unet2D_tanh(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=32):
        super(Unet2D_tanh, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel


        self.down1 = StackEncoder_tanh(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder_tanh(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder_tanh(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)
        self.down4 = StackEncoder_tanh(self.start_channel*4, self.start_channel*8, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBntanh(self.start_channel*8, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBntanh(self.start_channel*16, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder_tanh(in_channels=self.start_channel*16, out_channels=self.start_channel*8, padding=padding, momentum=momentum)
        self.up2 = StackDecoder_tanh(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up3 = StackDecoder_tanh(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up4 = StackDecoder_tanh(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        center = self.center(x)

        x = self.up1(center, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)

        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        if out.shape[-1] != self.width:
            out = self.output_up_seg_map(out)

        return out,center





class ae_lung_tanh(nn.Module):
    # down 4
    def __init__(self,in_shape,start_channel=32,kenel_size=3,padding=1,advers_mode=False):
        super(ae_lung_tanh, self).__init__()

        self.channels, self.heights, self.width = in_shape
        self.advers_mode = advers_mode
        self.encoder = nn.Sequential(
            nn.Conv2d(1, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 2, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*4),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 4, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),

            # center
            nn.Conv2d(start_channel * 8, start_channel * 16, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 16),
            nn.Tanh(),



        )

        self.decoder = nn.Sequential(


            nn.ConvTranspose2d(start_channel*16, start_channel*16, 2, stride=2),
            nn.Conv2d(start_channel * 16, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*8),
            nn.Tanh(),

            nn.ConvTranspose2d(start_channel * 8, start_channel * 8, 2, stride=2),
            nn.Conv2d(start_channel * 8, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.Tanh(),

            nn.ConvTranspose2d(start_channel * 4, start_channel*4, 2, stride=2),
            nn.Conv2d(start_channel * 4, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.Tanh(),

            nn.ConvTranspose2d(start_channel*2 , start_channel*2, 2, stride=2),
            nn.Conv2d(start_channel * 2, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.Tanh(),



            nn.Conv2d(start_channel, 1, 1)

        )
    def _random_sampling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        #t=torch.distributions.normal.Normal(channel_mean, channel_std,True)

        #import ipdb;ipdb.set_trace()
        #t = torch.cat((channel_mean, channel_std), dim=2)
        return [channel_mean,channel_std]

    def forward(self, x):
        encoder = self.encoder(x)
        random = self._random_sampling(encoder)
        decoder = self.decoder(encoder)

        if self.advers_mode:
            return decoder,encoder,random
        else:
            return decoder, encoder


# adversarial network
"""
https://github.com/xiehousen/OCGAN-Pytorch/blob/master/ocgan/networks.py
"""

class netlocalD(nn.Module):
    def __init__(self,start_channel=512):
        super(netlocalD, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 16 x 16
            nn.Conv2d(start_channel, start_channel//2, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf) x 8 x 8
            nn.Conv2d(start_channel//2,start_channel//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(start_channel//4),
            nn.ReLU(),
            # state size. (ndf*2) x 4 x 4
            nn.Conv2d(start_channel//4, start_channel//8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(start_channel//8),
            nn.ReLU(),
            # state size. (ndf*4) x 2 x 2
            nn.Conv2d(start_channel//8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):

        output = self.main(input)

        return output.view(-1, 1)




##########################################################################

# 05/14 model-multiple embedding loss



class Unet2D_multipleE(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=32,multipleE=1):
        super(Unet2D_multipleE, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel
        self.multipleE = multipleE


        self.down1 = StackEncoder(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)
        self.down4 = StackEncoder(self.start_channel*4, self.start_channel*8, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*8, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(self.start_channel*16, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=self.start_channel*16, out_channels=self.start_channel*8, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        center = self.center(x)

        x = self.up1(center, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)

        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)


        if self.multipleE ==1 :
            return out,x_trace1
        elif self.multipleE == 2:
            return out, [x_trace4,center]
        elif self.multipleE == 3:
            return out, [x_trace3,x_trace4,center]
        elif self.multipleE == 4:
            return out, [x_trace2,x_trace3,x_trace4,center]


class ae_lung_multipleE(nn.Module):
    # down 4
    def __init__(self,in_shape,start_channel=32,kenel_size=3,padding=1,multipleE=1):
        super(ae_lung_multipleE, self).__init__()

        self.channels, self.heights, self.width = in_shape
        self.multipleE = multipleE

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),

        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 2),
            nn.ReLU(),

        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 2, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),


        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 4, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.ReLU(),

        )
        self.center = nn.Sequential(

            nn.MaxPool2d(2, stride=2),
            #center
            nn.Conv2d(start_channel * 8, start_channel * 16, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 16),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(start_channel*16, start_channel*16, 2, stride=2),
            nn.Conv2d(start_channel * 16, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*8),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 8, start_channel * 8, 2, stride=2),
            nn.Conv2d(start_channel * 8, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 4, start_channel*4, 2, stride=2),
            nn.Conv2d(start_channel * 4, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel*2 , start_channel*2, 2, stride=2),
            nn.Conv2d(start_channel * 2, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),



            nn.Conv2d(start_channel, 1, 1)

        )


    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        center = self.center(encoder4)

        decoder = self.decoder(center)

        if self.multipleE == 1:
            return decoder, encoder1
        elif self.multipleE == 2:
            return decoder, [encoder4, center]
        elif self.multipleE == 3:
            return decoder, [encoder3, encoder4, center]
        elif self.multipleE == 4:
            return decoder, [encoder2, encoder3, encoder4, center]


#########################################################################################################################

class ConvBnRelu_Shared(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(ConvBnRelu_Shared, self).__init__()
        self.padding = padding
        self.stride = stride

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, stride=stride, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x, we="None"):
        if we == "None" :
            weights = []
            x = self.conv(x)
            weights.append(self.conv.weight)
            x = self.bn(x)
            x = self.relu(x)

        else:
            x = F.conv_transpose2d(input=x, weight=we, padding=self.padding, stride=self.stride)
            x = self.bn(x)
            x = self.relu(x)
            weights = "None"


        return x, weights


class Shared_StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(Shared_StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu_Shared(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding)

        self.maxPool = nn.Conv2d(out_channels,out_channels,kernel_size=(2, 2), stride=2, bias=False)

    def forward(self, x):
        weights = []
        x, conv_weight = self.convr1(x)
        weights.append(conv_weight[0])
        x = self.maxPool(x)
        weights.append(self.maxPool.weight)

        return x, weights


class Shared_StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(Shared_StackDecoder, self).__init__()

        self.convr1 = ConvBnRelu_Shared(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)


    def forward(self, x, weights="None"):
        x= F.conv_transpose2d(input=x, weight=weights[-1], stride=2)
        x, _ = self.convr1(x, weights[0])

        return x




class ae_lung_shared(nn.Module):
    # down 4
    def __init__(self,in_shape,start_channel=32,kenel_size=3,padding=1):
        super(ae_lung_shared, self).__init__()

        self.channels, self.heights, self.width = in_shape
        self.start_channel = start_channel

        self.encoder1 = Shared_StackEncoder(self.channels, self.start_channel, padding)
        self.encoder2 = Shared_StackEncoder(self.start_channel, self.start_channel*2, padding)
        self.encoder3 = Shared_StackEncoder(self.start_channel*2, self.start_channel*4, padding)
        self.encoder4 = Shared_StackEncoder(self.start_channel*4, self.start_channel*8, padding)

        self.center = nn.Sequential(
            #center
            nn.Conv2d(start_channel * 8, start_channel * 16, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 16),
            nn.ReLU())

        self.center1 = nn.Sequential(
            # center
            nn.Conv2d(start_channel * 16, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.ReLU())

        self.decoder1 = Shared_StackDecoder(self.start_channel*8, self.start_channel*4, padding)
        self.decoder2 = Shared_StackDecoder(self.start_channel*4, self.start_channel*2, padding)
        self.decoder3 = Shared_StackDecoder(self.start_channel*2, self.start_channel, padding)
        self.decoder4 = Shared_StackDecoder(self.start_channel, self.channels, padding)




    def forward(self, x):

        x,we1 = self.encoder1(x)
        #print('encoder1 output : ',x.shape)
        x,we2 = self.encoder2(x)
        #print('encoder2 output : ', x.shape)
        x,we3 = self.encoder3(x)
        #print('encoder3 output : ', x.shape)
        x,we4 = self.encoder4(x)
        #print('encoder4 output : ', x.shape)

        center = self.center(x)
        x = self.center1(center)
        #print('center output : ', x.shape)


        x = self.decoder1(x,we4)
        #print('decoder1 output : ', x.shape)
        x = self.decoder2(x,we3)
        #print('decoder2 output : ', x.shape)
        x = self.decoder3(x,we2)
        #print('decoder3 output : ', x.shape)
        x = self.decoder4(x,we1)
        #print('decoder4 output : ', x.shape)


        return x, center




##########################################################################

# class GramMatrix(nn.Module):
#
#     def forward(self, input):
#         a, b, c, d = input.size()  # a=batch size(=1)
#         features = input.view(a*b , -1)
#         G = torch.mm(features, features.t())  # compute the gram product
#
#         return G.div(a * b * c * d)


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a,b ,c*d)
        features_t = features.transpose(1, 2)
        G = features.bmm(features_t)  # compute the gram product

        return G.div(b * c * d)

#
# def gram_matrix(y):
#     (b, ch, h, w) = y.size()
#     features = y.view(b, ch, w * h)
#     features_t = features.transpose(1, 2)
#     gram = features.bmm(features_t) / (ch * h * w)
#     return gram

class Unet2D_style(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=32,style=1):
        super(Unet2D_style, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel
        self.style = style
        self.gram_matrix1 = GramMatrix()
        self.gram_matrix2 = GramMatrix()
        self.gram_matrix3 = GramMatrix()
        self.gram_matrix4 = GramMatrix()
        self.gram_matrix5 = GramMatrix()


        self.down1 = StackEncoder(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)
        self.down4 = StackEncoder(self.start_channel*4, self.start_channel*8, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*8, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(self.start_channel*16, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=self.start_channel*16, out_channels=self.start_channel*8, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')



    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        center = self.center(x)

        x = self.up1(center, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)

        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)


        if self.style == 0 :
            gram1 = self.gram_matrix1(x_trace1)
            gram2 = self.gram_matrix2(x_trace2)
            gram3 = self.gram_matrix3(x_trace3)
            gram4 = self.gram_matrix4(x_trace4)
            gram5 = self.gram_matrix5(center)

            return out, [gram1,gram2,gram3,gram4,gram5]


        elif self.style == 1 :


            gram2 = self.gram_matrix2(x_trace2)
            gram3 = self.gram_matrix3(x_trace3)
            gram4 = self.gram_matrix4(x_trace4)
            gram5 = self.gram_matrix5(center)

            return out, [gram2,gram3,gram4,gram5] ,x_trace4

        elif self.style == 2 :


            gram3 = self.gram_matrix3(x_trace3)
            gram4 = self.gram_matrix4(x_trace4)
            gram5 = self.gram_matrix5(center)

            return out, [gram3,gram4,gram5]

        elif self.style == 3 :



            gram4 = self.gram_matrix4(x_trace4)
            gram5 = self.gram_matrix5(center)

            return out, [gram4,gram5]

        elif self.style == 4 :

            gram5 = self.gram_matrix5(center)

            return out, [gram5]




class ae_lung_style(nn.Module):
    # down 4
    def __init__(self,in_shape,start_channel=32,kenel_size=3,padding=1,style=1):
        super(ae_lung_style, self).__init__()

        self.channels, self.heights, self.width = in_shape
        self.style = style
        self.gram_matrix1 = GramMatrix()
        self.gram_matrix2 = GramMatrix()
        self.gram_matrix3 = GramMatrix()
        self.gram_matrix4 = GramMatrix()
        self.gram_matrix5 = GramMatrix()



        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),

        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 2),
            nn.ReLU(),

        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            CoordConv(start_channel*2, start_channel*2,with_r = True, kernel_size =kenel_size, padding= padding),
            nn.Conv2d(start_channel * 2, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),


        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            CoordConv(start_channel * 4, start_channel * 4,with_r = True, kernel_size=kenel_size, padding=padding),
            nn.Conv2d(start_channel * 4, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.ReLU(),

        )
        self.center = nn.Sequential(

            nn.MaxPool2d(2, stride=2),
            #center
            CoordConv(start_channel * 8, start_channel * 8, with_r=True, kernel_size=kenel_size, padding=padding),
            nn.Conv2d(start_channel * 8, start_channel * 16, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 16),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(start_channel*16, start_channel*16, 2, stride=2),
            nn.Conv2d(start_channel * 16, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*8),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 8, start_channel * 8, 2, stride=2),
            nn.Conv2d(start_channel * 8, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 4, start_channel*4, 2, stride=2),
            nn.Conv2d(start_channel * 4, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel*2 , start_channel*2, 2, stride=2),
            nn.Conv2d(start_channel * 2, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),



            nn.Conv2d(start_channel, 1, 1)

        )



    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        center = self.center(encoder4)

        decoder = self.decoder(center)

        if self.style == 0 :

            gram1 = self.gram_matrix1(encoder1)
            gram2 = self.gram_matrix2(encoder2)
            gram3 = self.gram_matrix3(encoder3)
            gram4 = self.gram_matrix4(encoder4)
            gram5 = self.gram_matrix5(center)

            return decoder, [gram1,gram2,gram3,gram4,gram5]

        elif self.style == 1:

            gram2 = self.gram_matrix2(encoder2)
            gram3 = self.gram_matrix3(encoder3)
            gram4 = self.gram_matrix4(encoder4)
            gram5 = self.gram_matrix5(center)

            return decoder, [gram2, gram3, gram4,gram5], encoder4

        elif self.style == 2 :


            gram3 = self.gram_matrix3(encoder3)
            gram4 = self.gram_matrix4(encoder4)
            gram5 = self.gram_matrix5(center)

            return decoder, [ gram3, gram4,gram5]

        elif self.style == 3 :

            gram4 = self.gram_matrix4(encoder4)
            gram5 = self.gram_matrix5(center)

            return decoder, [gram4,gram5]

        elif self.style == 4 :

            gram5 = self.gram_matrix5(center)

            return decoder, [ gram5]








##########################################################################
if __name__ == '__main__':
    from torchsummary import summary

    my_net = ae_lung_style(in_shape=(1, 256, 256))
    summary(model=my_net.cuda(), input_size=(1, 256, 256))



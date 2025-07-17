#!/usr/bin/env python
# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from lct import LCT



class space_channel_attention(nn.Module):

    def __init__(self, channel, groups=64):
        super(space_channel_attention, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class CTMAM(nn.Module):

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(CTMAM, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=8, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=8, padding=(0, 1))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=128, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.sca = space_channel_attention(128)
        self.lstm_mfcc = nn.LSTM(input_size=6, hidden_size=3, num_layers=2, batch_first=True, dropout=0.25,
                                 bidirectional=True)

        self.LCT = LCT(img_size=(6, 14), num_classes=4, embed_dims=128, fc_dim=1280,
                                 num_heads=2, mlp_ratios=4, qkv_bias=True, qk_scale=None, representation_size=None,
                                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                 norm_layer=None,
                                 depths=3, qk_ratio=1, sr_ratios=2, dp=0.1, noclass=None)


    def forward(self, *input):

        #CNN BLOCK
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)
        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxp(x)#32*12*28
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)#64*6*14
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        #T-SA Attention
        xp=x.permute(0, 2,3,1)
        pool=nn.AdaptiveAvgPool2d((14,1))
        xp=pool(xp)
        xp=xp.squeeze(3)
        xp=xp.permute(0,2,1)
        xp,_=self.lstm_mfcc(xp)
        xp=xp.sigmoid()
        xp=xp.permute(0,2,1).unsqueeze(1)
        x=x*xp
        x = self.sca(x)#128,128,6,14

        #lCT BLOCK
        x=self.LCT(x)

        return x

#emodb数据集
class CTMAM_EMODB(nn.Module):

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(CTMAM_EMODB, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=8, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=8, padding=(0, 1))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=128, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.sca = space_channel_attention(128)


        self.lstm_mfcc = nn.LSTM(input_size=6, hidden_size=3, num_layers=2, batch_first=True, dropout=0.25,
                                 bidirectional=True)
        self.lct = LCT(img_size=(6, 14), num_classes=7, embed_dims=128,fc_dim=1280,
                              num_heads=1, mlp_ratios=4, qkv_bias=True, qk_scale=None, representation_size=None,
                              drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                              depths=1, qk_ratio=1, sr_ratios=2, dp=0.1, noclass=None)


    def forward(self, *input):
        # CNN BLOCK
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)
        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxp(x)#32*12*28
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)#64*6*14
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        # T-SA Attention
        xp=x.permute(0, 2,3,1)
        pool=nn.AdaptiveAvgPool2d((14,1))
        xp=pool(xp)
        xp=xp.squeeze(3)
        xp=xp.permute(0,2,1)
        xp,_=self.lstm_mfcc(xp)
        xp=xp.sigmoid()
        xp=xp.permute(0,2,1).unsqueeze(1)
        x=x*xp
        x = self.sca(x)#128,128,6,14

        # lCT BLOCK
        x=self.lct(x)

        return x


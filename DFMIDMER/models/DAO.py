from __future__ import absolute_import

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class DAO(nn.Module):
    def __init__(self, in_channel, **kwargs):
        super(DAO, self).__init__()
        self.k = 8

        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        conv, conv1, W = [], [], []
        for _ in range(2):
            conv.append(nn.Conv3d(in_channel, 1, 1))
            conv1.append(ConvBlock(self.k, self.k, 1))

            add_block = nn.BatchNorm3d(in_channel)
            nn.init.constant_(add_block.weight.data, 0.0)
            nn.init.constant_(add_block.bias.data, 0.0)
            W.append(add_block)

        add_block = nn.BatchNorm3d(in_channel)
        nn.init.constant_(add_block.weight.data, 0.0)
        nn.init.constant_(add_block.bias.data, 0.0)
        W.append(add_block)

        add_block = nn.BatchNorm3d(in_channel)
        nn.init.constant_(add_block.weight.data, 0.0)
        nn.init.constant_(add_block.bias.data, 0.0)
        W.append(add_block)

        self.conv = nn.ModuleList(conv)
        self.conv1 = nn.ModuleList(conv1)
        self.W = nn.ModuleList(W)
        print(self.W)
           
    def forward_conv(self, x, i):
        """x: [b, c, t, h, w]
        """
        b, c, t, h, w = x.size()
        # print('h=', h, ' w=', w)
        assert (h == self.k)
        assert (w == 1)

        y = self.conv[i](x)
        y = y.view(b, t, h * w)
        y = y.transpose(1, 2).contiguous() 
        s = self.conv1[i](y.unsqueeze(-1).unsqueeze(-1)) 
        s = s.transpose(1, 2).contiguous() 
        s = torch.softmax(s, 2)
        a = s.view(b * t, h * w)

        x = x.view(b, c, t, -1) 
        s = s.view(b, 1, t, -1)
        y = (x * s).sum(-1)

        y = self.W[i + 2](y.unsqueeze(-1).unsqueeze(-1))
        return y, a

    def forward_avg(self, x, i):
        """x: [b, c, t, h, w]
        """
        b, c, t, h, w = x.size()
        # print('h=', h, ' w=', w)
        assert (h * w == self.k)
        x_in = x

        x = x.mean(1) #[b, t, h, w]
        x = x.view(b, t, -1)
        s = torch.softmax(x, -1)
        # print('forward_avg  after-softmax  s.shape', s.shape)
        a = s.view(b * t, h * w)
        # print('forward_avg  a.shape', a.shape)

        y = x_in.view(b, c, t, -1)
        # print('forward_avg  y.shape', y.shape)
        s = s.unsqueeze(1)
        # print('forward_avg  s.shape', s.shape)
        y = (y * s).sum(-1)
        # print('forward_avg  y.shape', y.shape)
        y = y.unsqueeze(-1).unsqueeze(-1)
        # print('forward_avg  y.shape', y.shape)
        y = self.W[i](y)
        # print('forward_avg  final  y.shape', y.shape)

        return y, a

    def forward(self, x1, x2):
        """
        x1: [bs, c, 2, 16, 8]
        x2: [bs, c, 6, 8, 4]
        """
        b, c, t1, h1, w1 = x1.size()
        b, c, t2, h2, w2 = x2.size()
        x1_in, x2_in = x1, x2

        if h1 != h2:
            x1 = self.pool(x1)
        x1 = x1.mean(-1, keepdim=True)
        x2 = x2.mean(-1, keepdim=True)  # [b, C, 14, 8, 1]

        x2_1 = torch.stack((x2[:, :, 0], x2[:, :, 3]), 2)
        x2_2 = torch.stack((x2[:, :, 1], x2[:, :, 4]), 2)
        x2_3 = torch.stack((x2[:, :, 2], x2[:, :, 5]), 2)
        # x2_3 = torch.stack((x2[:, :, 2], x2[:, :, 9]), 2)
        # x2_4 = torch.stack((x2[:, :, 3], x2[:, :, 10]), 2)
        # x2_5 = torch.stack((x2[:, :, 4], x2[:, :, 11]), 2)
        # x2_6 = torch.stack((x2[:, :, 5], x2[:, :, 12]), 2)
        # x2_7 = torch.stack((x2[:, :, 6], x2[:, :, 13]), 2)

        y1, a1 = self.forward_avg(x1, 0)
        y2_1, a2_1 = self.forward_avg(x2_1, 1)
        y2_2, a2_2 = self.forward_conv(x2_2, 0)
        y2_3, a2_3 = self.forward_conv(x2_3, 1)
        # y2_3, a2_3 = self.forward_conv(x2_3, 1)
        # y2_4, a2_4 = self.forward_conv(x2_4, 1)
        # y2_5, a2_5 = self.forward_conv(x2_5, 1)
        # y2_6, a2_6 = self.forward_conv(x2_6, 1)
        # y2_7, a2_7 = self.forward_conv(x2_7, 1)
        # y2 = torch.stack((y2_1[:, :, 0], y2_2[:, :, 0], y2_3[:, :, 0], y2_4[:, :, 0], y2_5[:, :, 0], y2_6[:, :, 0], y2_7[:, :, 0],
        #                   y2_1[:, :, 1], y2_2[:, :, 1], y2_3[:, :, 1], y2_4[:, :, 1], y2_5[:, :, 1], y2_6[:, :, 1], y2_7[:, :, 1]), 2)  # [b, c, 14, 1, 1]
        y2 = torch.stack((y2_1[:, :, 0], y2_2[:, :, 0], y2_3[:, :, 0], y2_1[:, :, 1], y2_2[:, :, 1], y2_3[:, :, 1]), 2)

        # print('y1.shape', y1.shape, 'x1_in.shape', x1_in.shape)
        z1 = y1 + x1_in
        # print('y2.shape', y2.shape, 'x2_in.shape', x2_in.shape)
        z2 = y2 + x2_in

        return z1, z2, [a2_1, a2_2, a2_3]

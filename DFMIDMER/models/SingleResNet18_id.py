from __future__ import absolute_import

import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from models import inflate
from models.resnets1 import resnet50_s1
from models.resnets1 import resnet18
import torch


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=False)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1,
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward_once(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

    def forward(self, x):

        out = self.forward_once(x)
        return out


class Basicblock3d(nn.Module):

    def __init__(self, basicblock2d):
        super(Basicblock3d, self).__init__()

        self.conv1 = inflate.inflate_conv(basicblock2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(basicblock2d.bn1)
        self.conv2 = inflate.inflate_conv(basicblock2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(basicblock2d.bn2)
        # self.conv3 = inflate.inflate_conv(basicblock2d.conv3, time_dim=1)
        # self.bn3 = inflate.inflate_batch_norm(basicblock2d.bn3)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=False)

        if basicblock2d.downsample is not None:
            self.downsample = self._inflate_downsample(basicblock2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
            downsample3d = nn.Sequential(
                inflate.inflate_conv(downsample2d[0], time_dim=1,
                                     time_stride=time_stride),
                inflate.inflate_batch_norm(downsample2d[1]))
            return downsample3d

    def forward_once(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            # out = self.relu(out)

            # basicblock does not have conv3
            # out = self.conv3(out)
            # out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
            return out

    def forward(self, x):

            out = self.forward_once(x)
            return out


class SingleResNet18_id(nn.Module):

    def __init__(self, num_classes, **kwargs):

        super(SingleResNet18_id, self).__init__()
        resnet2d = resnet18(pretrained=True)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=False)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.downsample = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.layer1 = self._inflate_reslayer(resnet2d.layer1)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4)

        # fc using random initialization
        # add_block = nn.BatchNorm1d(2048)
        add_block = nn.BatchNorm1d(512)
        add_block.apply(weights_init_kaiming)
        self.bn = add_block

        # classifier using Random initialization
        # classifier = nn.Linear(2048, num_classes)
        classifier = nn.Linear(512, num_classes)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def _inflate_reslayer(self, reslayer2d):
        reslayers3d = []
        for layer2d in reslayer2d:
            # layer3d = Bottleneck3d(layer2d)
            layer3d = Basicblock3d(layer2d)
            reslayers3d.append(layer3d)

        return nn.Sequential(*reslayers3d)

    def pooling(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('x1', x.shape)
        x = self.maxpool(x)
        # print('x2', x.shape)

        # xh = x
        # xl = x
        # xl = self.downsample(xl)  # [b, c, K, h//2, w//2]
        x = self.downsample(x)
        # print('x3', x.shape)

        # layer1
        x = self.layer1(x)
        # print('x4', x.shape)

        # layer2
        x = self.layer2(x)
        # print('x5', x.shape)

        # layer3
        x = self.layer3(x)
        # print('x6', x.shape)

        # layer4
        x = self.layer4(x)
        # print('x7', x.shape)

        # xh = self.pooling(xh)  # [bs, 2, c]
        # xl = self.pooling(xl)  # [bs, K, c]
        x = self.pooling(x)
        # print('x8', x.shape)

        # xh = xh.mean(1, keepdims=True)  # [b, 1, c]
        # xl = xl.mean(1, keepdims=True)  # [b, 1, c]
        x = x.mean(1, keepdims=True)
        # print('x9', x.shape)

        x = x.mean(1)
        # print('x10', x.shape)
        f = self.bn(x)
        y = self.classifier(f)

        return y, x


if __name__ == '__main__':
    model = SingleResNet18_id(num_classes=6)
    x = torch.ones([2, 3, 16, 224, 224])
    print(model)
    print('model(x)', model(x))
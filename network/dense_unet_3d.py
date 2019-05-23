from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)))


class DenseNet3D(nn.Module):
    def __init__(self, in_ch, growth_rate=36, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet3D, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_ch, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        return features


class DenseUNet3D(nn.Module):
    def __init__(self, in_ch, out_ch=3):
        super(DenseUNet3D, self).__init__()
        densenet3d = DenseNet3D(in_ch)
        backbone = list(list(densenet3d.children())[0].children())

        self.conv1 = nn.Sequential(*backbone[:3])
        self.mp = backbone[3]
        self.denseblock1 = backbone[4]
        self.transition1 = backbone[5]
        self.denseblock2 = backbone[6]
        self.transition2 = backbone[7]
        self.denseblock3 = backbone[8]
        self.transition3 = backbone[9]
        self.denseblock4 = backbone[10]
        self.bn = backbone[11]
        self.up1 = _Up(x1_ch=1659, x2_ch=1590, out_ch=504)
        self.up2 = _Up(x1_ch=504, x2_ch=588, out_ch=224)
        self.up3 = _Up(x1_ch=224, x2_ch=312, out_ch=192)
        self.up4 = _Up(x1_ch=192, x2_ch=96, out_ch=96, scale_factor=(2, 2, 2))
        self.up5 = nn.Sequential(
            _Interpolate(scale_factor=(2, 2, 2)),
            nn.BatchNorm3d(num_features=96),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=96, out_channels=64, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x_ = self.mp(x)
        x1 = self.denseblock1(x_)
        x1t = self.transition1(x1)
        x2 = self.denseblock2(x1t)
        x2t = self.transition2(x2)
        x3 = self.denseblock3(x2t)
        x3t = self.transition3(x3)
        x4 = self.denseblock4(x3t)
        x4 = self.bn(x4)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.up4(x7, x)
        feat = self.up5(x8)
        cls = self.conv2(feat)
        return feat, cls


class _Interpolate(nn.Module):
    def __init__(self, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True):
        super(_Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                                      align_corners=self.align_corners)
        return x


class _Up(nn.Module):
    def __init__(self, x1_ch, x2_ch, out_ch, scale_factor=(2, 2, 1)):
        super(_Up, self).__init__()
        self.up = _Interpolate(scale_factor=scale_factor)
        self.conv1x1 = nn.Conv3d(in_channels=x2_ch, out_channels=x1_ch, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=x1_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv1x1(x2)
        x = x1 + x2
        x = self.conv(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    net = DenseUNet3D(in_ch=4).cuda()
    summary(net, (4, 224, 224, 12))

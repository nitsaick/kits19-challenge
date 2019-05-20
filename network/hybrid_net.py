import torch.nn as nn
import torch.nn.functional as F

from .dense_unet_3d import DenseNet3D


class HybridFeatureFusionLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HybridFeatureFusionLayer, self).__init__()
        self.hff = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=64, kernel_size=3, padding=1),
            nn.Dropout3d(p=0.3),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=out_ch, kernel_size=1)
        )

    def forward(self, feat_2d, feat3d):
        x = feat_2d + feat3d
        x = self.hff(x)
        return x


class HybridNet(nn.Module):
    def __init__(self, in_ch):
        super(HybridNet, self).__init__()
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
        self.up1 = _Up(x1_ch=504, x2_ch=496, out_ch=504)
        self.up2 = _Up(x1_ch=504, x2_ch=224, out_ch=224)
        self.up3 = _Up(x1_ch=224, x2_ch=192, out_ch=192)
        self.up4 = _Up(x1_ch=192, x2_ch=96, out_ch=96, scale_factor=(2, 2, 2))
        self.up5 = nn.Sequential(
            _Interpolate(scale_factor=(2, 2, 2)),
            nn.BatchNorm3d(num_features=96),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=96, out_channels=64, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=3, kernel_size=1)

        self.hff = HybridFeatureFusionLayer(in_ch=64, out_ch=3)

    def forward(self, x, feat_2d):
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
        feat_3d = self.up5(x8)
        # cls = self.conv2(feat)
        cls = self.hff(feat_2d, feat_3d)
        return cls


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
            nn.BatchNorm3d(num_features=x1_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=x1_ch, out_channels=out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv1x1(x2)
        x = x1 + x2
        x = self.conv(x)
        return x

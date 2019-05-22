import torch.nn as nn
from torchvision import models


class DenseUNet2D(nn.Module):
    def __init__(self):
        super(DenseUNet2D, self).__init__()
        densenet = models.densenet161(pretrained=True)
        backbone = list(list(densenet.children())[0].children())

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
        self.up1 = _Up(x1_ch=2208, x2_ch=2112, out_ch=768)
        self.up2 = _Up(x1_ch=768, x2_ch=768, out_ch=384)
        self.up3 = _Up(x1_ch=384, x2_ch=384, out_ch=96)
        self.up4 = _Up(x1_ch=96, x2_ch=96, out_ch=96)
        self.up5 = nn.Sequential(
            _Interpolate(),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

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
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=True):
        super(_Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                                      align_corners=self.align_corners)
        return x


class _Up(nn.Module):
    def __init__(self, x1_ch, x2_ch, out_ch):
        super(_Up, self).__init__()
        self.up = _Interpolate()
        self.conv1x1 = nn.Conv2d(in_channels=x2_ch, out_channels=x1_ch, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=x1_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
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
    net = DenseUNet2D().cuda()
    summary(net, (3, 224, 224))

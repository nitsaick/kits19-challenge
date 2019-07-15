import functools

import torch.nn as nn


class Discriminator(nn.Module):
    
    def __init__(self, in_ch=3, dim=64, n_downsamplings=4):
        super().__init__()
        
        Norm = functools.partial(nn.InstanceNorm2d, affine=True)
        
        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )
        
        layers = []
        
        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(in_ch, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        
        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))
        
        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        y = self.net(x)
        return y


if __name__ == '__main__':
    from torchsummary import summary
    
    net = Discriminator(in_ch=3).cuda()
    summary(net, (3, 224, 224))

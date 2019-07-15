import torch
import torch.nn as nn

from loss.util import class2one_hot


class WGanLoss(nn.Module):
    def __init__(self, gradient_penalty=True, weight=10.0):
        super(WGanLoss, self).__init__()
        self.gradient_penalty = gradient_penalty
        self.gradient_penalty_weight = weight
    
    def forward(self, imgs, labels, outputs, discriminator, num_classes, type):
        if type == 'd':
            outputs = outputs.detach()
        
        labels = class2one_hot(labels, num_classes).type(torch.float32)
        
        real_imgs = torch.cat([imgs, labels], dim=1)
        real_outputs = discriminator(real_imgs)
        
        fake_imgs = torch.cat([imgs, outputs], dim=1)
        fake_outputs = discriminator(fake_imgs)
        
        if type == 'd':
            dis_loss = -real_outputs.mean() + fake_outputs.mean()
            if self.gradient_penalty:
                dis_loss += self._gradient_penalty(discriminator, real_imgs, fake_imgs)
            return dis_loss
        
        elif type == 'g':
            gen_loss = -fake_outputs.mean()
            return gen_loss
        
        else:
            raise ValueError
    
    def _gradient_penalty(self, discriminator, labels, outputs):
        shape = [labels.size(0)] + [1] * (labels.dim() - 1)
        alpha = torch.rand(shape, device=labels.device)
        interpolate = (labels + alpha * (outputs - labels)).detach()
        interpolate.requires_grad = True
        pred = discriminator(interpolate)
        grad = torch.autograd.grad(pred, interpolate, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
        norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)
        gradient_penalty = ((norm - 1) ** 2).mean()
        
        return gradient_penalty * self.gradient_penalty_weight

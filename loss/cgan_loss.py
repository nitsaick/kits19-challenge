import torch
import torch.nn as nn

from loss.util import class2one_hot


class CGanLoss(nn.Module):
    def __init__(self):
        super(CGanLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
    
    def forward(self, imgs, labels, outputs, discriminator, num_classes, type):
        if type == 'd':
            outputs = outputs.detach()
            
        labels = class2one_hot(labels, num_classes).type(torch.float32)
        
        real_imgs = torch.cat([imgs, labels], dim=1)
        real_outputs = discriminator(real_imgs)
        real_labels = torch.ones_like(real_outputs)
        
        fake_imgs = torch.cat([imgs, outputs], dim=1)
        fake_outputs = discriminator(fake_imgs)
        fake_labels = torch.zeros_like(fake_outputs)
        
        if type == 'd':
            dis_loss = (self.l2_loss(real_outputs, real_labels) + self.l2_loss(fake_outputs, fake_labels)) * 0.5
            return dis_loss
        
        elif type == 'g':
            gen_loss = self.l2_loss(fake_outputs, real_labels)
            return gen_loss
        
        else:
            raise ValueError

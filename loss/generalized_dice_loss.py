import torch
import torch.nn as nn
from torch import Tensor, einsum

from loss.util import simplex


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(GeneralizedDiceLoss, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs['idc']
        # print(f'Initialized {self.__class__.__name__} with {kwargs}')
    
    def forward(self, probs: Tensor, target: Tensor):
        assert simplex(probs) and simplex(target)
        
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        
        w = 1 / ((einsum('bc...->bc', tc).type(torch.float32) + 1e-10) ** 2)
        
        if len(pc.shape) == 4:
            intersection = w * einsum('bchw,bchw->bc', pc, tc)
        elif len(pc.shape) == 5:
            intersection = w * einsum('bchwd,bchwd->bc', pc, tc)
        
        union = w * (einsum('bc...->bc', pc) + einsum('bc...->bc', tc))
        
        divided = 1 - (2 * intersection + 1e-10) / (union + 1e-10)
        
        loss = divided.mean()
        
        return loss

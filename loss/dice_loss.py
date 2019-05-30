import torch
import torch.nn as nn
from torch import Tensor, einsum

from loss.util import simplex


class DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs['idc']
        print(f'Initialized {self.__class__.__name__} with {kwargs}')

    def forward(self, probs: Tensor, target: Tensor):
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection= einsum('bcwh,bcwh->bc', pc, tc)
        union = (einsum('bcwh->bc', pc) + einsum('bcwh->bc', tc))

        divided = 1 - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss

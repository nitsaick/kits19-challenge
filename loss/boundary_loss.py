import torch
import torch.nn as nn
from torch import Tensor, einsum

from loss.util import simplex, one_hot


class BoundaryLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BoundaryLoss, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs['idc']
        print(f'Initialized {self.__class__.__name__} with {kwargs}')

    def forward(self, probs: Tensor, dist_maps: Tensor):
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum('bcwh,bcwh->bcwh', pc, dc)

        loss = multipled.mean()

        return loss

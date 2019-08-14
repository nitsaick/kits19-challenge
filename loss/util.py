from typing import Iterable, Set

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance
from torch import Tensor


# check sum of probability in channel axis is 1
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def one_hot(t: Tensor, axis=1) -> bool:
    a = simplex(t, axis)
    b = sset(t, [0, 1])
    return simplex(t, axis) and sset(t, [0, 1])


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)
    
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
    
    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert one_hot(res)
    
    return res


def np_class2one_hot(seg: np.ndarray, C: int) -> np.ndarray:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = np.expand_dims(seg, axis=0)
    
    res = np.stack([seg == c for c in range(C)], axis=1).astype(np.int32)
    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)
    
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)
    
    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=1)
    C: int = seg.shape[1]
    
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[:, c].astype(np.bool)
        
        if posmask.any():
            negmask = ~posmask
            res[:, c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    
    return res


if __name__ == '__main__':
    a = np.load('../data/case_00000/segmentation/296.npy')
    b = np_class2one_hot(a, 3)
    c = one_hot2dist(b)[0]
    d = b[0].copy()
    d[0, ...] = 0
    d[1, ...] = 0
    d[2, ...] = 1
    
    multipled = np.einsum('cwh,cwh->cwh', d, c)
    
    loss = multipled.mean()
    
    d = (c + 437) / 3.44 / 255
    from utils.vis import imshow
    
    # imshow('b', b[0].transpose((1, 2, 0)) * 128, (1, 1))
    # imshow('c', c.transpose((1, 2, 0)), (1, 1))
    imshow('c0', c[0], (1, 1))
    imshow('c1', c[1], (1, 1))
    imshow('c2', c[2], (1, 1))
    # imshow('d', d.transpose((1, 2, 0)), (1, 1))
    ...

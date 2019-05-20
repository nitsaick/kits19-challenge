import glob
import os

import torch


def save(epoch, net, optimizer, root):
    torch.save({'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, root)


def find_latest(root):
    files = glob.glob(os.path.join(root, '*.pth'))
    if len(files):
        latest_file = max(files, key=os.path.getctime)
        return latest_file
    else:
        raise FileNotFoundError('No checkpoint file in "{}"'.format(root))


def load_params(net=None, optimizer=None, device='cpu', root=None, latest=False):
    if latest:
        root = find_latest(root)
    assert os.path.isfile(root)

    checkpoint = torch.load(root, map_location=device)
    epoch = checkpoint['epoch'] + 1
    if net is not None:
        net.load_state_dict(checkpoint['net'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return net, optimizer, epoch

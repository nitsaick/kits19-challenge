import torch
from pathlib2 import Path


def save(epoch, net, optimizer, root):
    torch.save({'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, root)


def _key_exist(data, cp, key):
    return key in data and data[key] and key in cp and cp[key]


def load_params(data, cp_file, device='cpu'):
    cp_file = Path(cp_file)
    assert cp_file.exists()
    
    cp = torch.load(str(cp_file), map_location=device)
    if _key_exist(data, cp, key='net'):
        data['net'].load_state_dict(cp['net'])
    
    if _key_exist(data, cp, key='optimizer'):
        data['optimizer'].load_state_dict(cp['optimizer'])
        for state in data['optimizer'].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    if _key_exist(data, cp, key='epoch'):
        data['epoch'] = cp['epoch']
    
    return data

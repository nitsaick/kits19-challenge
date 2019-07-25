import json

import click
import cv2
import numpy as np
import torch
from pathlib2 import Path
from tqdm import tqdm

import utils.checkpoint as cp
from network import ResUNet


@click.command()
@click.option('-r', '--resume', help='esume checkpoint',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), required=True)
@click.option('--data', 'data_path', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('-o', '--output', 'roi_file', help='output roi file path',
              type=click.Path(file_okay=True, resolve_path=True), default='roi.json', show_default=True)
def get_coarse_roi(resume, data_path, roi_file):
    data_path = Path(data_path)
    stack_num = 5
    net = ResUNet(in_ch=stack_num, out_ch=2, base_ch=64)
    
    cp_file = Path(resume)
    net, _, _ = cp.load_params(net, None, root=str(cp_file))
    
    torch.cuda.empty_cache()
    
    # to GPU device
    net = net.cuda()
    
    net.eval()
    torch.set_grad_enabled(False)
    
    test_case_file = Path(data_path) / 'test.txt'
    test_case = []
    f = open(test_case_file, 'r')
    for line in f:
        test_case.append(int(line))
    
    min_x = min_y = min_z = 10000
    max_x = max_y = max_z = -1
    
    rois = {}
    roi_file = Path(roi_file)
    
    for case in tqdm(test_case):
        case_root = data_path / f'case_{case:05d}'
        imaging_dir = case_root / 'imaging'
        case_imgs = sorted(list(imaging_dir.glob('*.npy')))
        
        vol = []
        for idx in range(len(case_imgs)):
            imgs = []
            for i in range(idx - stack_num // 2, idx + stack_num // 2 + 1):
                if i < 0:
                    i = 0
                elif i >= len(case_imgs):
                    i = len(case_imgs) - 1
                img_path = case_imgs[i]
                img = np.load(str(img_path))
                imgs.append(img)
            
            imgs = np.stack(imgs, axis=0)
            imgs = imgs.astype(np.float32)
            imgs = torch.from_numpy(imgs)
            imgs = torch.unsqueeze(imgs, dim=0)
            
            imgs = imgs.cuda()
            outputs = net(imgs)
            outputs = outputs.argmax(dim=1)
            outputs = outputs.cpu().detach().numpy()
            vol.append(outputs)
        
        vol = np.concatenate(vol, axis=0)
        vol = vol.astype(np.uint8)
        
        for i, img in enumerate(vol):
            if img.max() != 0:
                rect = cv2.boundingRect(img)
                min_x = min(min_x, rect[0])
                min_y = min(min_y, rect[1])
                min_z = min(min_z, i)
                max_x = max(max_x, rect[0] + rect[2])
                max_y = max(max_y, rect[1] + rect[3])
                max_z = max(max_z, i)
        
        roi = {'min_x': min_x, 'min_y': min_y, 'min_z': min_z, 'max_x': max_x, 'max_y': max_y, 'max_z': max_z}
        
        case_roi = {'kidney': roi}
        rois.update({f'case_{case:05d}': case_roi})
        
        with open(roi_file, 'w') as f:
            json.dump(rois, f, indent=4, separators=(',', ': '))
        
        min_x = min_y = min_z = 10000
        max_x = max_y = max_z = -1


if __name__ == '__main__':
    get_coarse_roi()

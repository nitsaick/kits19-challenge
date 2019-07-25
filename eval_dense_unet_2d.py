import json

import cc3d
import click
import nibabel as nib
import numpy as np
import torch
from pathlib2 import Path
from tqdm import tqdm

import utils.checkpoint as cp
from network import DenseUNet2D

import cv2

@click.command()
@click.option('--data', 'data_path', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('-r', '--resume', help='Resume checkpoint file to continue training',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), default=None, required=True)
@click.option('-o', '--output', 'output_path', help='output image path',
              type=click.Path(dir_okay=True, resolve_path=True), default='out', show_default=True)
def main(data_path, resume, output_path):
    # prepare
    data_path = Path(data_path)
    org_data_path = Path('D:/Qsync/workspace/kits19/data')
    stack_num = 3
    net = DenseUNet2D(out_ch=3)
    
    cp_file = Path(resume)
    net, _, _ = cp.load_params(net, None, root=str(cp_file))
    
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    torch.cuda.empty_cache()
    
    # to GPU device
    net = net.cuda()
    
    net.eval()
    torch.set_grad_enabled(False)
    
    roi_path = Path('roi.json')
    with open(roi_path, 'r') as f:
        rois = json.load(f)
    roi_d = 2
    roi_range = 10
    
    test_case_file = Path(data_path) / 'test.txt'
    test_case = []
    f = open(test_case_file, 'r')
    for line in f:
        test_case.append(int(line))
    
    for case in tqdm(test_case):
        case_root = data_path / f'case_{case:05d}'
        imaging_dir = case_root / 'imaging'
        case_imgs = sorted(list(imaging_dir.glob('*.npy')))
        slices = len(case_imgs)
        
        roi = rois[f'case_{case:05d}']['kidney']
        min_z = max(0, roi['min_z'] - roi_d)
        max_z = min(slices, roi['max_z'] + roi_d + 1)
        case_imgs = case_imgs[min_z:max_z]
        
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
                min_y = max(0, roi['min_y'] - roi_range)
                max_y = min(img.shape[0], roi['max_y'] + roi_range + 1)
                min_x = max(0, roi['min_x'] - roi_range)
                max_x = min(img.shape[1], roi['max_x'] + roi_range + 1)
                mask = np.ones_like(img, dtype=np.bool)
                mask[min_y:max_y, min_x:max_x] = False
                img[mask] = 0
                imgs.append(img)
            
            imgs = np.stack(imgs, axis=0)
            imgs = imgs.astype(np.float32)
            imgs = torch.from_numpy(imgs)
            imgs = torch.unsqueeze(imgs, dim=0)
            
            imgs = imgs.cuda()
            _, outputs, _, _, _, _ = net(imgs)
            outputs = outputs.argmax(dim=1)
            outputs = outputs.cpu().detach().numpy()
            vol.append(outputs)
        
        vol_min_z = []
        for _ in range(0, min_z):
            vol_min_z.append(np.zeros_like(outputs))
        vol_max_z = []
        for _ in range(max_z, slices):
            vol_max_z.append(np.zeros_like(outputs))
        vol = vol_min_z + vol + vol_max_z
        
        vol = np.concatenate(vol, axis=0)
        vol = vol.astype(np.uint8)
        
        org_data = org_data_path / f'case_{case:05d}' / 'imaging.nii.gz'
        affine = nib.load(str(org_data)).get_affine()
        
        vol_nii = nib.Nifti1Image(vol, affine=affine)
        vol_nii_filename = output_path / f'prediction_{case:05d}_o.nii.gz'
        vol_nii.to_filename(str(vol_nii_filename))
        
        vol_ = vol.copy()
        vol_[vol_ > 0] = 1
        vol_cc = cc3d.connected_components(vol_)
        cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
        cc_sum.sort(key=lambda x: x[1], reverse=True)
        cc_sum.pop(0)  # remove background
        reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * 0.1]
        for i in reduce_cc:
            vol[vol_cc == i] = 0
        
        vol_nii = nib.Nifti1Image(vol, affine=affine)
        vol_nii_filename = output_path / f'prediction_{case:05d}.nii.gz'
        vol_nii.to_filename(str(vol_nii_filename))


if __name__ == '__main__':
    main()

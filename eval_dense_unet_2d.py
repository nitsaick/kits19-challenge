import gzip
import shutil

import click
import nibabel as nib
import numpy as np
import torch
from pathlib2 import Path
from tqdm import tqdm

import utils.checkpoint as cp
from network import DenseUNet2D

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
    org_data_path = Path('D:/Nick/Downloads/kits19/data')
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
    
    test_case_file = Path(data_path) / 'test.txt'
    test_case = []
    f = open(test_case_file, 'r')
    for line in f:
        test_case.append(int(line))
    
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
            _, outputs, _, _, _, _ = net(imgs)
            outputs = outputs.argmax(dim=1)
            outputs = outputs.cpu().detach().numpy()
            vol.append(outputs)
        
        vol = np.concatenate(vol, axis=0)
        vol = vol.astype(np.uint8)
        org_data = org_data_path / f'case_{case:05d}' / 'imaging.nii.gz'
        affine = nib.load(str(org_data)).get_affine()
        vol_nii = nib.Nifti1Image(vol, affine=affine)
        vol_nii_filename = output_path / f'prediction_{case:05d}.nii'
        vol_niigz_filename = output_path / f'prediction_{case:05d}.nii.gz'
        vol_nii.to_filename(str(vol_nii_filename))
        with open(str(vol_nii_filename), 'rb') as f_in, gzip.open(str(vol_niigz_filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    main()

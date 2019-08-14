import json

import click
import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from pathlib2 import Path
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS19
from dataset.transform import MedicalTransform
from network import ResUNet
from utils.vis import imshow


@click.command()
@click.option('-d', '--data', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True), required=True)
@click.option('-o', '--output', 'roi_file', help='output roi file path',
              type=click.Path(file_okay=True, resolve_path=True), default='roi_gt.json', show_default=True)
def get_roi_from_gt(data, roi_file):
    data = Path(data)
    
    cases = sorted([d for d in data.iterdir() if d.is_dir()])
    case_idx = 0
    rois = {}
    for case in tqdm(cases):
        img_file = case / 'imaging.nii.gz'
        assert img_file.exists()
        img = nib.load(str(img_file)).get_data()
        total_z, total_y, total_x = img.shape
        vol = {'total_x': total_x, 'total_y': total_y, 'total_z': total_z}
        case_data = {'vol': vol}
        
        seg_file = case / 'segmentation.nii.gz'
        if seg_file.exists():
            seg = nib.load(str(seg_file)).get_data()
            kidney = calc(seg, idx=1)
            tumor = calc(seg, idx=2)
            case_data.update({'kidney': kidney, 'tumor': tumor})
        
        rois[f'case_{case_idx:05d}'] = case_data
        
        with open(roi_file, 'w') as f:
            json.dump(rois, f, indent=4, separators=(',', ': '))
        
        case_idx += 1


def calc(seg, idx):
    bincount = np.bincount(seg.flatten())
    area = int(bincount[idx])
    
    value = []
    for i in range(seg.shape[0]):
        value.append(seg[i].max())
    value = np.array(value)
    
    slice_ = np.where(value > idx - 1)[0]
    num_slice = len(slice_)
    min_z = int(slice_.min())
    max_z = int(slice_.max()) + 1
    min_x = min_y = 10000
    max_x = max_y = -1
    for i in range(min_z, max_z):
        if seg[i].max() > idx - 1:
            mask = np.ma.masked_where(seg[i] > idx - 1, seg[i]).mask
            rect = cv2.boundingRect(mask.astype(np.uint8))
            min_x = min(min_x, rect[0])
            min_y = min(min_y, rect[1])
            max_x = max(max_x, rect[0] + rect[2])
            max_y = max(max_y, rect[1] + rect[3])
    
    roi = {'min_x': min_x, 'min_y': min_y, 'min_z': min_z,
           'max_x': max_x, 'max_y': max_y, 'max_z': max_z,
           'area': area, 'slice': num_slice}
    
    return roi


@click.command()
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=1, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('-s', '--size', 'img_size', help='Output image size', type=(int, int),
              default=(512, 512), show_default=True)
@click.option('--data', 'data_path', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('-r', '--resume', help='Resume model',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), required=True)
@click.option('-o', '--output', 'roi_file', help='output roi file path',
              type=click.Path(file_okay=True, resolve_path=True), default='roi_gt.json', show_default=True)
@click.option('--vis_intvl', help='Number of iteration interval of display visualize image. '
                                  'No display when set to 0',
              type=int, default=20, show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
def get_roi_from_resunet(batch_size, num_gpu, img_size, data_path, resume, roi_file, vis_intvl, num_workers):
    with open(roi_file, 'r') as f:
        rois = json.load(f)
        
    data_path = Path(data_path)
    
    transform = MedicalTransform(output_size=img_size, roi_error_range=15, use_roi=False)
    
    dataset = KiTS19(data_path, stack_num=5, spec_classes=[0, 1, 1], img_size=img_size,
                     use_roi=False, train_transform=transform, valid_transform=transform)
    
    net = ResUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=64)
    
    if resume:
        data = {'net': net}
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')
    
    gpu_ids = [i for i in range(num_gpu)]
    
    torch.cuda.empty_cache()
    
    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()
    
    net.eval()
    torch.set_grad_enabled(False)
    transform.eval()
    
    subset = dataset.test_dataset
    case_slice_indices = dataset.test_case_slice_indices
    
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)
    
    case = 0
    vol_output = []
    
    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/test', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, idx = data['image'].cuda(), data['index']
            
            predicts = net(imgs)
            predicts = predicts.argmax(dim=1)
            
            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()
            
            vol_output.append(predicts)
            
            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]
                
                vol = vol_output[:vol_num_slice]
                kidney = calc(vol, idx=1)
                case_roi = {'kidney': kidney}
                case_id = dataset.case_idx_to_case_id(case, 'test')
                rois[f'case_{case_id:05d}'].update(case_roi)
                with open(roi_file, 'w') as f:
                    json.dump(rois, f, indent=4, separators=(',', ': '))
                
                vol_output = [vol_output[vol_num_slice:]]
                case += 1
                pbar.update(1)
            
            if vis_intvl > 0 and batch_idx % vis_intvl == 0:
                data['predict'] = predicts
                data = dataset.vis_transform(data)
                imgs, predicts = data['image'], data['predict']
                imshow(title=f'eval/test', imgs=(imgs[0, 1], predicts[0]), shape=(1, 2),
                       subtitle=('image', 'predict'))


if __name__ == '__main__':
    get_roi_from_gt()
    get_roi_from_resunet()

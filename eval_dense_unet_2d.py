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
from network import DenseUNet2D
from utils.vis import imshow


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
@click.option('-o', '--output', 'output_path', help='output image path',
              type=click.Path(dir_okay=True, resolve_path=True), default='out', show_default=True)
@click.option('--vis_intvl', help='Number of iteration interval of display visualize image. '
                                  'No display when set to 0',
              type=int, default=20, show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
def main(batch_size, num_gpu, img_size, data_path, resume, output_path, vis_intvl, num_workers):
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    roi_error_range = 15
    transform = MedicalTransform(output_size=img_size, roi_error_range=roi_error_range, use_roi=True)
    
    dataset = KiTS19(data_path, stack_num=3, spec_classes=[0, 1, 2], img_size=img_size,
                     use_roi=True, roi_file='roi.json', roi_error_range=5, test_transform=transform)
    
    net = DenseUNet2D(in_ch=dataset.img_channels, out_ch=dataset.num_classes)
    
    if resume:
        data = {'net': net}
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')
    
    gpu_ids = [i for i in range(num_gpu)]
    
    print(f'{" Start evaluation ":-^40s}\n')
    msg = f'Net: {net.__class__.__name__}\n' + \
          f'Dataset: {dataset.__class__.__name__}\n' + \
          f'Batch size: {batch_size}\n' + \
          f'Device: cuda{str(gpu_ids)}\n'
    print(msg)
    
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
            
            outputs = net(imgs)
            predicts = outputs['output']
            predicts = predicts.argmax(dim=1)
            
            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()
            
            vol_output.append(predicts)
            
            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]
                
                roi = dataset.get_roi(case, type='test')
                vol = vol_output[:vol_num_slice]
                vol_ = reverse_transform(vol, roi, dataset, transform)
                vol_ = vol_.astype(np.uint8)
                
                case_id = dataset.case_idx_to_case_id(case, type='test')
                affine = np.load(data_path / f'case_{case_id:05d}' / 'affine.npy')
                vol_nii = nib.Nifti1Image(vol_, affine)
                vol_nii_filename = output_path / f'prediction_{case_id:05d}.nii.gz'
                vol_nii.to_filename(str(vol_nii_filename))
                
                vol_output = [vol_output[vol_num_slice:]]
                case += 1
                pbar.update(1)
            
            if vis_intvl > 0 and batch_idx % vis_intvl == 0:
                data['predict'] = predicts
                data = dataset.vis_transform(data)
                imgs, predicts = data['image'], data['predict']
                imshow(title=f'eval/test', imgs=(imgs[0, 1], predicts[0]), shape=(1, 2),
                       subtitle=('image', 'predict'))


def reverse_transform(vol, roi, dataset, transform):
    min_x = max(0, roi['kidney']['min_x'] - transform.roi_error_range)
    max_x = min(vol.shape[-1], roi['kidney']['max_x'] + transform.roi_error_range)
    min_y = max(0, roi['kidney']['min_y'] - transform.roi_error_range)
    max_y = min(vol.shape[-2], roi['kidney']['max_y'] + transform.roi_error_range)
    min_z = max(0, roi['kidney']['min_z'] - dataset.roi_error_range)
    max_z = min(roi['vol']['total_z'], roi['kidney']['max_z'] + dataset.roi_error_range)
    
    min_height = roi['vol']['total_y']
    min_width = roi['vol']['total_x']
    
    roi_rows = max_y - min_y
    roi_cols = max_x - min_x
    max_size = max(transform.output_size[0], transform.output_size[1])
    scale = max_size / float(max(roi_cols, roi_rows))
    rows = int(roi_rows * scale)
    cols = int(roi_cols * scale)
    
    if rows < min_height:
        h_pad_top = int((min_height - rows) / 2.0)
        h_pad_bottom = rows + h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = min_height
    
    if cols < min_width:
        w_pad_left = int((min_width - cols) / 2.0)
        w_pad_right = cols + w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = min_width
    
    for i in range(len(vol)):
        img = vol[i]
        reverse_padding_img = img[h_pad_top:h_pad_bottom, w_pad_left:w_pad_right]
        reverse_padding_img = reverse_padding_img.astype(np.uint8)
        reverse_resize_img = cv2.resize(reverse_padding_img, dsize=(max_x - min_x, max_y - min_y),
                                        interpolation=cv2.INTER_LINEAR)
        reverse_resize_img = reverse_resize_img.astype(np.int64)
        reverse_img = np.zeros((min_height, min_width))
        reverse_img[min_y:max_y, min_x: max_x] = reverse_resize_img
        vol[i] = reverse_img
    
    size = (1, min_height, min_width)
    vol_min_z = [np.zeros(size) for _ in range(0, min_z)]
    vol_max_z = [np.zeros(size) for _ in range(max_z, roi['vol']['total_z'])]
    
    vol = vol_min_z + [vol] + vol_max_z
    vol = np.concatenate(vol, axis=0)
    
    assert vol.shape == (roi['vol']['total_z'], roi['vol']['total_y'], roi['vol']['total_x'])
    
    return vol


if __name__ == '__main__':
    main()

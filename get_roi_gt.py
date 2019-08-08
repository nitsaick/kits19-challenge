import json

import click
import cv2
import nibabel as nib
import numpy as np
from pathlib2 import Path
from tqdm import tqdm


@click.command()
@click.option('-d', '--data', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True), required=True)
@click.option('-o', '--output', 'roi_file', help='output roi file path',
              type=click.Path(file_okay=True, resolve_path=True), default='roi_gt.json', show_default=True)
def get_roi_gt(data, roi_file):
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
            
            _, k_area, t_area = np.bincount(seg.flatten())
            k_area, t_area = int(k_area), int(t_area)
            
            value = []
            for i in range(seg.shape[0]):
                value.append(seg[i].max())
            value = np.array(value)
            kidney = np.where(value > 0)[0]
            tumor = np.where(value > 1)[0]
            k_slice, t_slice = len(kidney), len(tumor)
            
            k_min_z = int(kidney.min())
            k_max_z = int(kidney.max()) + 1
            t_min_z = int(tumor.min())
            t_max_z = int(tumor.max()) + 1
            
            k_min_x = k_min_y = t_min_x = t_min_y = 10000
            k_max_x = k_max_y = t_max_x = t_max_y = -1
            for i in range(k_min_z, k_max_z):
                if seg[i].max() > 0:
                    rect = cv2.boundingRect(seg[i].astype(np.uint8))
                    k_min_x = min(k_min_x, rect[0])
                    k_min_y = min(k_min_y, rect[1])
                    k_max_x = max(k_max_x, rect[0] + rect[2]) + 1
                    k_max_y = max(k_max_y, rect[1] + rect[3]) + 1
                    if seg[i].max() > 1:
                        tumor_mask = np.ma.masked_where(seg[i] == 2, seg[i]).mask
                        rect = cv2.boundingRect(tumor_mask.astype(np.uint8))
                        t_min_x = min(t_min_x, rect[0])
                        t_min_y = min(t_min_y, rect[1])
                        t_max_x = max(t_max_x, rect[0] + rect[2]) + 1
                        t_max_y = max(t_max_y, rect[1] + rect[3]) + 1
            
            k_roi = {'min_x': k_min_x, 'min_y': k_min_y, 'min_z': k_min_z,
                     'max_x': k_max_x, 'max_y': k_max_y, 'max_z': k_max_z,
                     'area': k_area, 'slice': k_slice}
            t_roi = {'min_x': t_min_x, 'min_y': t_min_y, 'min_z': t_min_z,
                     'max_x': t_max_x, 'max_y': t_max_y, 'max_z': t_max_z,
                     'area': t_area, 'slice': t_slice}
            
            case_data.update({'kidney': k_roi, 'tumor': t_roi})
        rois[f'case_{case_idx:05d}'] = case_data
        
        with open(roi_file, 'w') as f:
            json.dump(rois, f, indent=4, separators=(',', ': '))
        
        case_idx += 1


if __name__ == '__main__':
    get_roi_gt()

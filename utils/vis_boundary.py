import cv2
import numpy as np
from pathlib2 import Path


def vis_boundary(img, label, pred, num_classes):
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    label = label.astype(np.uint8)
    pred = pred.astype(np.uint8)
    
    imgs = []
    for cls in range(1, num_classes):
        label_boundary = np.zeros_like(label)
        pred_boundary = np.zeros_like(pred)
        
        label_boundary[label == cls] = label[label == cls]
        pred_boundary[pred == cls] = pred[pred == cls]
        
        label_boundary = cv2.Canny(label_boundary, 0, 1)
        pred_boundary = cv2.Canny(pred_boundary, 0, 1)
        # overlay = np.zeros_like(label)
        # overlay[np.logical_and(label_boundary, pred_boundary)] = 255
        
        img_boindary = img.copy()
        img_boindary[pred_boundary == 255] = [255, 0, 0]
        img_boindary[label_boundary == 255] = [0, 255, 0]
        # img_boindary[overlay == 255] = [0, 0, 255]
        
        imgs.append(img_boindary)
    
    return imgs


if __name__ == '__main__':
    from utils.vis import imshow
    
    root = Path('data')
    cases = sorted([d for d in root.iterdir() if d.is_dir()])
    for case in cases:
        img_path = case / 'imaging'
        seg_path = case / 'segmentation'
        num_slice = len(list(img_path.glob('*.npy')))
        for i in range(num_slice // 2, num_slice // 3 * 2):
            img_file = img_path / f'{i:03d}.npy'
            seg_file = seg_path / f'{i:03d}.npy'
            img = np.load(str(img_file))
            seg = np.load(str(seg_file))
            vis_img = vis_boundary(img, seg, seg, 3)
            imshow('vis1', vis_img[0])
            imshow('vis2', vis_img[1])

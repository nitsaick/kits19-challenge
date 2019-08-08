import json
from pathlib import Path

import cv2
import numpy as np
import torch
from albumentations import (
    PadIfNeeded,
    Compose,
    Resize
)
from torch.utils import data

from dataset.transform import to_numpy


class KiTS19(data.Dataset):
    def __init__(self, root, stack_num=1, spec_classes=None, img_size=(512, 512),
                 train_case_ids_file='train.txt', valid_case_ids_file='val.txt', test_case_ids_file='test.txt',
                 use_roi=False, roi_file=None, roi_error_range=0,
                 train_transform=None, valid_transform=None, test_transform=None):
        self._root = Path(root)
        self._stack_num = stack_num
        if spec_classes is None:
            self._spec_classes = [0, 1, 2]
        else:
            assert len(self.get_classes_name(spec=False)) == len(spec_classes)
            self._spec_classes = spec_classes
        
        self._img_size = img_size
        
        self._use_roi = use_roi
        if use_roi:
            self._rois = None
            _roi_file = self._root / roi_file
            assert _roi_file.exists()
            with open(_roi_file, 'r') as f:
                self._rois = json.load(f)
            self._roi_error_range = roi_error_range
        
        self._train_transform = train_transform
        self._valid_transform = valid_transform
        self._test_transform = test_transform
        
        self._get_data(train_case_ids_file, valid_case_ids_file, test_case_ids_file)
        self._split_subset()
        
        self._num_classes = len(self.get_classes_name())
        self._img_channels = self.__getitem__(0)['image'].shape[0]
    
    def _get_data(self, train_case_ids_file, valid_case_ids_file, test_case_ids_file):
        def read_txt(file):
            d = []
            f = open(file, 'r')
            for line in f:
                d.append(int(line))
            return d
        
        train_case_ids_file = self._root / train_case_ids_file
        valid_case_ids_file = self._root / valid_case_ids_file
        test_case_ids_file = self._root / test_case_ids_file
        self._train_case = read_txt(train_case_ids_file)
        self._valid_case = read_txt(valid_case_ids_file)
        self._test_case = read_txt(test_case_ids_file)
        self._case_id = self._train_case + self._valid_case + self._test_case
        
        train_imgs, train_labels, train_case_slice_num = self._read_npy(self._root, self._train_case, is_test=False)
        valid_imgs, valid_labels, valid_case_slice_num = self._read_npy(self._root, self._valid_case, is_test=False)
        test_imgs, test_labels, test_case_slice_num = self._read_npy(self._root, self._test_case, is_test=True)
        
        self._imgs = train_imgs + valid_imgs + test_imgs
        self._labels = train_labels + valid_labels + test_labels
        
        self._indices = list(range(len(self._imgs)))
        self._train_indices = self._indices[:len(train_imgs)]
        self._valid_indices = self._indices[len(train_imgs):len(train_imgs) + len(valid_imgs)]
        self._test_indices = self._indices[
                             len(train_imgs) + len(valid_imgs): len(train_imgs) + len(valid_imgs) + len(test_imgs)]
        
        idx = 0
        self._case_slice_indices = [0]
        self._train_case_slice_indices = [0]
        for num in train_case_slice_num:
            idx += num
            self._case_slice_indices.append(idx)
            self._train_case_slice_indices.append(idx)
        
        self._valid_case_slice_indices = [self._train_case_slice_indices[-1]]
        for num in valid_case_slice_num:
            idx += num
            self._case_slice_indices.append(idx)
            self._valid_case_slice_indices.append(idx)
        
        self._test_case_slice_indices = [self._valid_case_slice_indices[-1]]
        for num in test_case_slice_num:
            idx += num
            self._case_slice_indices.append(idx)
            self._test_case_slice_indices.append(idx)
    
    def _read_npy(self, root, cases, is_test=False):
        imgs = []
        labels = []
        case_slice_num = []
        
        for case in cases:
            case_root = root / f'case_{case:05d}'
            imaging_dir = case_root / 'imaging'
            assert imaging_dir.exists()
            case_imgs = sorted(list(imaging_dir.glob('*.npy')))
            
            min_z = 0
            max_z = len(case_imgs)
            if self._use_roi:
                roi = self._rois[f'case_{case:05d}']['kidney']
                min_z = max(min_z, roi['min_z'] - self._roi_error_range)
                max_z = min(max_z, roi['max_z'] + self._roi_error_range)
            
            case_imgs = case_imgs[min_z: max_z]
            imgs += case_imgs
            
            if not is_test:
                segmentation_dir = case_root / 'segmentation'
                assert segmentation_dir.exists()
                case_labels = sorted(list(segmentation_dir.glob('*.npy')))
                case_labels = case_labels[min_z: max_z]
                labels += case_labels
                assert len(imgs) == len(labels)
            
            case_slice_num.append(len(case_imgs))
        
        return imgs, labels, case_slice_num
    
    def _split_subset(self):
        self._train_dataset = data.Subset(self, self._train_indices)
        self._valid_dataset = data.Subset(self, self._valid_indices)
        self._test_dataset = data.Subset(self, self._test_indices)
    
    def get_classes_name(self, spec=True):
        classes_name = np.array(['background', 'kidney', 'tumor'])
        
        if not spec:
            return classes_name
        
        spec_classes_name = []
        for i in classes_name[self._spec_classes]:
            if i not in spec_classes_name:
                spec_classes_name.append(i)
        return spec_classes_name
    
    def get_colormap(self, spec=True):
        cmap = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
        cmap = np.array(cmap, dtype=np.int)
        
        if not spec:
            return cmap
        
        spec_cmap = []
        for i in cmap[self._spec_classes]:
            if len(spec_cmap) == 0:
                spec_cmap.append(i)
            else:
                duplicate = False
                for j in spec_cmap:
                    duplicate = duplicate or (i == j).all()
                if not duplicate:
                    spec_cmap.append(i)
        return np.array(spec_cmap)
    
    def idx_to_name(self, idx):
        path = self._imgs[idx]
        name = Path(path.parts[-3]) / Path(path.parts[-1][:-4])
        return name
    
    def vis_transform(self, data):
        cmap = self.get_colormap()
        if 'image' in data.keys() and data['image'] is not None:
            imgs = data['image']
            if type(imgs).__module__ != np.__name__:
                imgs = imgs.cpu().detach().numpy()
            data['image'] = imgs
        
        if 'label' in data.keys() and data['label'] is not None:
            labels = data['label']
            if type(labels).__module__ != np.__name__:
                labels = labels.cpu().detach().numpy()
            labels = cmap[labels]
            labels = labels.transpose((0, 3, 1, 2))
            labels = labels / 255
            data['label'] = labels
        
        if 'predict' in data.keys() and data['predict'] is not None:
            preds = data['predict']
            if type(preds).__module__ != np.__name__:
                preds = preds.cpu().detach().numpy()
            if preds.shape[1] == self.num_classes:
                preds = preds.argmax(axis=1)
            preds = cmap[preds]
            preds = preds.transpose((0, 3, 1, 2))
            preds = preds / 255
            data['predict'] = preds
        
        return data
    
    def _default_transform(self, data):
        if (data['image'].shape[0], data['image'].shape[1]) != self._img_size:
            data = self._resize(data)
        
        image, label = data['image'], data['label']
        
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        data['image'] = image
        
        if label is not None:
            label = label.astype(np.int64)
            
            if self._spec_classes != [0, 1, 2]:
                idx = list(range(len(self.get_classes_name(spec=False))))
                masks = [np.where(label == i) for i in idx]
                spec_class_idx = []
                for i in self._spec_classes:
                    if i not in spec_class_idx:
                        spec_class_idx.append(i)
                
                for mask, spec_class in zip(masks, self._spec_classes):
                    label[mask] = spec_class_idx.index(spec_class)
            
            label = torch.from_numpy(label)
            data['label'] = label
        
        return data
    
    @staticmethod
    def normalize(vol):
        hu_max = 512
        hu_min = -512
        vol = np.clip(vol, hu_min, hu_max)
    
        mxval = np.max(vol)
        mnval = np.min(vol)
        volume_norm = (vol - mnval) / max(mxval - mnval, 1e-3)
    
        return volume_norm
    
    def _resize(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']
        
        num = max(img.shape[0], img.shape[1])
        
        aug = Compose([
            PadIfNeeded(min_height=num, min_width=num,
                        border_mode=cv2.BORDER_CONSTANT, p=1),
            Resize(height=self._img_size[0], width=self._img_size[1], p=1)
        ])
        
        data = aug(image=img, mask=label)
        img, label = data['image'], data['mask']
        
        data['image'] = img
        data['label'] = label
        return data
    
    def img_idx_to_case_idx(self, idx):
        case_idx = 0
        for i in range(len(self._case_slice_indices) - 1):
            if self._case_slice_indices[i] <= idx < self._case_slice_indices[i + 1]:
                case_idx = i
                break
        return case_idx
    
    def case_idx_to_case_id(self, case_idx):
        return self._case_id[case_idx]
    
    def get_stack_img(self, idx):
        case_idx = self.img_idx_to_case_idx(idx)
        imgs = []
        for i in range(idx - self._stack_num // 2, idx + self._stack_num // 2 + 1):
            if i < self._case_slice_indices[case_idx]:
                i = self._case_slice_indices[case_idx]
            elif i >= self._case_slice_indices[case_idx + 1]:
                i = self._case_slice_indices[case_idx + 1] - 1
            img_path = self._imgs[i]
            img = np.load(str(img_path))
            imgs.append(img)
        img = np.stack(imgs, axis=2)
        
        if idx in self._test_indices:
            label = None
        else:
            label_path = self._labels[idx]
            label = np.load(str(label_path))
        
        roi = self.get_roi(case_idx)
        data = {'image': img, 'label': label, 'index': idx, 'roi': roi}
        
        return data
    
    def get_roi(self, case_idx):
        case_id = self.case_idx_to_case_id(case_idx)
        roi = self._rois[f'case_{case_id:05d}']['kidney'] if self._use_roi else {}
        
        return roi
    
    def __getitem__(self, idx):
        data = self.get_stack_img(idx)
        
        if idx in self._train_indices and self._train_transform is not None:
            data = self._train_transform(data)
        elif idx in self._valid_indices and self._valid_transform is not None:
            data = self._valid_transform(data)
        elif idx in self._test_indices and self._test_transform is not None:
            data = self._test_transform(data)
        
        data = self._default_transform(data)
        
        return data
    
    def __len__(self):
        return len(self._imgs)
    
    @property
    def img_channels(self):
        return self._img_channels
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def spec_classes(self):
        return self._spec_classes
    
    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def valid_dataset(self):
        return self._valid_dataset
    
    @property
    def test_dataset(self):
        return self._test_dataset
    
    @property
    def train_case_slice_indices(self):
        return self._train_case_slice_indices
    
    @property
    def valid_case_slice_indices(self):
        return self._valid_case_slice_indices
    
    @property
    def test_case_slice_indices(self):
        return self._test_case_slice_indices
    
    @property
    def train_case(self):
        return self._train_case
    
    @property
    def valid_case(self):
        return self._valid_case
    
    @property
    def test_case(self):
        return self._test_case


import click


@click.command()
@click.option('--data', 'data_path', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
def main(data_path):
    from dataset.transform import MedicalTransform
    from torch.utils.data import DataLoader, SequentialSampler
    from utils.vis import imshow
    
    root = Path(data_path)
    transform = MedicalTransform(output_size=512, roi_error_range=15, use_roi=True)
    transform.eval()
    dataset = KiTS19(root, stack_num=3, spec_classes=[0, 1, 2], img_size=(512, 512),
                     use_roi=True, roi_file='roi.json', roi_error_range=5,
                     train_transform=transform, valid_transform=None, test_transform=None)
    
    subset = dataset.train_dataset
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=1, sampler=sampler)
    
    for batch_idx, data in enumerate(data_loader):
        data = dataset.vis_transform(data)
        imgs, labels = data['image'], data['label']
        imshow(title='KiTS19', imgs=(imgs[0][1], labels[0]))


if __name__ == '__main__':
    main()

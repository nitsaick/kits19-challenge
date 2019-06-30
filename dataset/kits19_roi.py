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


class KiTS19_roi(data.Dataset):
    def __init__(self, root, stack_num=1,
                 train_transform=None, valid_transform=None, spec_classes=None, roi_path=None, roi_d=5):
        self.root = Path(root)
        self.stack_num = stack_num
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        if roi_path is not None:
            self.roi_path = Path(roi_path)
        else:
            self.roi_path = self.root / 'roi.json'
        
        self.imgs, self.labels = self._get_img_list(self.root, roi_d)
        self._split_subset()
        
        if spec_classes is None:
            self._spec_classes = [0, 1, 2]
        else:
            assert len(self.get_classes_name(spec=False)) == len(spec_classes)
            self._spec_classes = spec_classes
        
        self._num_classes = len(self.get_classes_name())
        self._img_channels = self.__getitem__(0)[0].shape[0]
    
    def _get_img_list(self, root, roi_d):
        train_case_file = Path(root) / 'train.txt'
        valid_case_file = Path(root) / 'val.txt'
        
        self.train_case = []
        self.valid_case = []
        
        f = open(train_case_file, 'r')
        for line in f:
            self.train_case.append(int(line))
        
        f = open(valid_case_file, 'r')
        for line in f:
            self.valid_case.append(int(line))
            
        self.case = self.train_case + self.valid_case
        
        assert self.roi_path.exists()
        with open(self.roi_path, 'r') as f:
            self.rois = json.load(f)
        
        self.case_indices = [0, ]
        
        def read(root, cases, imgs, labels):
            for case in cases:
                case_root = root / f'case_{case:05d}'
                imaging_dir = case_root / 'imaging'
                segmentation_dir = case_root / 'segmentation'
                assert imaging_dir.exists() and segmentation_dir.exists()
                
                case_imgs = sorted(list(imaging_dir.glob('*.npy')))
                case_labels = sorted(list(segmentation_dir.glob('*.npy')))
                
                roi = self.rois[f'case_{case:05d}']['kidney']
                min_z = max(0, roi['min_z'] - roi_d)
                max_z = min(len(case_imgs) - 1, roi['max_z'] + roi_d)
                
                imgs += case_imgs[min_z:max_z + 1]
                labels += case_labels[min_z:max_z + 1]
                
                assert len(imgs) == len(labels)
                self.case_indices.append(len(imgs))
                
            return imgs, labels
        
        imgs = []
        labels = []
        
        read(root, self.train_case, imgs, labels)
        split = len(imgs)
        self.split_case = len(self.case_indices)

        read(root, self.valid_case, imgs, labels)
        
        self.indices = list(range(len(imgs)))
        self.train_indices = self.indices[:split]
        self.valid_indices = self.indices[split:]
        
        return imgs, labels
    
    def _split_subset(self):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        self.test_dataset = self
    
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
        path = self.imgs[idx]
        name = Path(path.parts[-3]) / Path(path.parts[-1][:-4])
        return name
    
    def vis_transform(self, imgs=None, labels=None, preds=None, to_plt=False):
        cmap = self.get_colormap()
        if imgs is not None:
            if type(imgs).__module__ != np.__name__:
                imgs = imgs.cpu().detach().numpy()
            if to_plt is True:
                imgs = imgs.transpose((0, 2, 3, 1))
        
        if labels is not None:
            if type(labels).__module__ != np.__name__:
                labels = labels.cpu().detach().numpy().astype('int')
            labels = cmap[labels]
            labels = labels.transpose((0, 3, 1, 2))
            if to_plt is True:
                labels = labels.transpose((0, 2, 3, 1))
            labels = labels / 255.
        
        if preds is not None:
            if type(preds).__module__ != np.__name__:
                preds = preds.cpu().detach().numpy()
            if preds.shape[1] == self.num_classes:
                preds = preds.argmax(axis=1)
            preds = cmap[preds]
            preds = preds.transpose((0, 3, 1, 2))
            if to_plt is True:
                preds = preds.transpose((0, 2, 3, 1))
            preds = preds / 255.
        
        return imgs, labels, preds
    
    def _default_transform(self, data):
        if data['image'].shape[0] != data['image'].shape[1] \
                and self.train_transform is None and self.valid_transform is None:
            data = kits19_transform()(data)
        
        image, label = data['image'], data['label']
        
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        label = label.astype(np.int64)
        
        idx = list(range(len(self.get_classes_name(spec=False))))
        masks = [np.where(label == i) for i in idx]
        
        if self._spec_classes != [0, 1, 2]:
            spec_class_idx = []
            for i in self._spec_classes:
                if i not in spec_class_idx:
                    spec_class_idx.append(i)
            
            for mask, spec_class in zip(masks, self._spec_classes):
                label[mask] = spec_class_idx.index(spec_class)
        
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return {'image': image, 'label': label}
    
    def __getitem__(self, idx):
        case_i = 0
        for i in range(len(self.case_indices) - 1):
            if self.case_indices[i] <= idx < self.case_indices[i + 1]:
                case_i = i
                break
        
        imgs = []
        for i in range(idx - self.stack_num // 2, idx + self.stack_num // 2 + 1):
            if i < self.case_indices[case_i]:
                i = self.case_indices[case_i]
            elif i >= self.case_indices[case_i + 1]:
                i = self.case_indices[case_i + 1] - 1
            img_path = self.imgs[i]
            img = np.load(str(img_path))
            imgs.append(img)
        
        roi = self.rois[f'case_{self.case[case_i]:05d}']['kidney']
        img = np.stack(imgs, axis=2)
        # img = img[roi['min_y']:roi['max_y'], roi['min_x']:roi['max_x'], :]
        
        label_path = self.labels[idx]
        label = np.load(str(label_path))
        # label = label[roi['min_y']:roi['max_y'], roi['min_x']:roi['max_x']]
        
        data = {'image': img, 'label': label, 'roi': roi}
        if idx in self.train_indices and self.train_transform is not None:
            data = self.train_transform(data)
        elif idx in self.valid_indices and self.valid_transform is not None:
            data = self.valid_transform(data)
        
        data = self._default_transform(data)
        img = data['image']
        label = data['label']
        
        return img, label, idx
    
    def __len__(self):
        return len(self.imgs)
    
    @property
    def img_channels(self):
        return self._img_channels
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def spec_classes(self):
        return self._spec_classes


class kits19_transform:
    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']
        
        num = max(img.shape[0], img.shape[1])
        
        aug = Compose([
            PadIfNeeded(min_height=num, min_width=num, border_mode=cv2.BORDER_CONSTANT, p=1),
            Resize(height=512, width=512, p=1)
        ])
        
        data = aug(image=img, mask=label)
        img, label = data['image'], data['mask']
        
        data = {'image': img, 'label': label}
        return data


if __name__ == '__main__':
    root = Path('H:\Qsync\workspace\kits19-challege\data')
    
    dataset = KiTS19_roi(root=root, stack_num=3,
                         train_transform=None,
                         valid_transform=None,
                         spec_classes=[0, 1, 2])
    
    from torch.utils.data import DataLoader, SequentialSampler
    from utils.vis import imshow
    
    subset = dataset.train_dataset
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=1, sampler=sampler)
    
    for batch_idx, (imgs, labels, idx) in enumerate(data_loader):
        img, label, _ = dataset.vis_transform(imgs=imgs, labels=labels, preds=None)
        imshow(title='kits19', imgs=(img[0][1], label[0]))

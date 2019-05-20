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


class kits19(data.Dataset):
    def __init__(self, root, stack_num=1, valid_rate=0.3,
                 train_transform=None, valid_transform=None, specified_classes=None):
        self.root = Path(root)
        self.imgs, self.labels = self.get_img_list(self.root, valid_rate)
        
        if specified_classes is None:
            self.specified_classes = [0, 1, 2]
        else:
            assert len(self.get_classes_name(spec=False)) == len(specified_classes)
            self.specified_classes = specified_classes
        
        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()
        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        
        self._split()
        
        self.stack_num = stack_num
        
        self.img_channels = self.__getitem__(0)[0].shape[0]
        self.vis_idx = [300, 264, 179, 188, 42, 333, 48, 40, 147, 45,
                        17, 19, 24, 40, 119, 41, 60, 37, 68, 42,
                        29, 25, 253, 43, 38, 63, 176, 366, 27, 76,
                        13, 41, 57, 172, 45, 40, 55, 32, 21, 37,
                        67, 20, 164, 39, 37, 44, 70, 78, 30, 350,
                        41, 31, 166, 135, 32, 27, 42, 58, 29, 192,
                        18, 14, 34, 194, 27, 67, 190, 164, 274, 23]
    
    def _split(self):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        self.test_dataset = self
    
    def get_img_list(self, root, valid_rate):
        imgs = []
        labels = []
        
        self.case_indices = [0, ]
        cases = sorted([d for d in root.iterdir() if d.is_dir()])
        self.split_case = int(np.round(len(cases) * valid_rate))
        for i in range(len(cases)):
            case = cases[i]
            imaging_dir = case / 'imaging'
            segmentation_dir = case / 'segmentation'
            assert imaging_dir.exists() and segmentation_dir.exists()
            
            imgs += sorted(list(imaging_dir.glob('*.npy')))
            labels += sorted(list(segmentation_dir.glob('*.npy')))
            
            assert len(imgs) == len(labels)
            self.case_indices.append(len(imgs))
            if case.stem[-3:] == f'{self.split_case:03}':
                split = len(imgs)
        
        self.indices = list(range(len(imgs)))
        self.train_indices = self.indices[split:]
        self.valid_indices = self.indices[:split]
        
        return imgs, labels
    
    def default_transform(self, data):
        if data['image'].shape[1] != data['image'].shape[2] \
                and self.train_transform is None and self.valid_transform is None:
            data = kits19_transform()(data)
        
        image, label = data['image'], data['label']
        
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        label = label.astype(np.int64)
        
        idx = list(range(len(self.get_classes_name(spec=False))))
        masks = [np.where(label == i) for i in idx]
        
        spec_class_idx = []
        
        if self.specified_classes != [0, 1, 2]:
            for i in self.specified_classes:
                if i not in spec_class_idx:
                    spec_class_idx.append(i)
            
            for mask, spec_class in zip(masks, self.specified_classes):
                label[mask] = spec_class_idx.index(spec_class)
        
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return {'image': image, 'label': label}
    
    def get_colormap(self, spec=True):
        cmap = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
        cmap = np.array(cmap, dtype=np.int)
        
        if not spec:
            return cmap
        else:
            spec_cmap = []
            for i in cmap[self.specified_classes]:
                if len(spec_cmap) == 0:
                    spec_cmap.append(i)
                else:
                    duplicate = False
                    for j in spec_cmap:
                        duplicate = duplicate or (i == j).all()
                    if not duplicate:
                        spec_cmap.append(i)
            return np.array(spec_cmap)
    
    def get_classes_name(self, spec=True):
        classes_name = np.array(['background', 'kidney', 'tumor'])
        
        if not spec:
            return classes_name
        else:
            spec_classes_name = []
            for i in classes_name[self.specified_classes]:
                if i not in spec_classes_name:
                    spec_classes_name.append(i)
            return spec_classes_name
    
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
    
    def __getitem__(self, idx):
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
        
        img = np.stack(imgs, axis=2)
        
        label_path = self.labels[idx]
        label = np.load(str(label_path))
        
        data = {'image': img, 'label': label}
        if idx in self.train_indices and self.train_transform is not None:
            data = self.train_transform(data)
        elif idx in self.valid_indices and self.valid_transform is not None:
            data = self.valid_transform(data)
        
        data = self.default_transform(data)
        
        img = data['image']
        label = data['label']
        
        return img, label, idx
    
    def __len__(self):
        return self.dataset_size


class kits19_roi(data.Dataset):
    def __init__(self, root, stack_num=1, valid_rate=0.3,
                 train_transform=None, valid_transform=None, specified_classes=None):
        self.root = Path(root)
        self.imgs, self.labels = self.get_img_list(self.root, valid_rate)
        
        if specified_classes is None:
            self.specified_classes = [0, 1, 2]
        else:
            assert len(self.get_classes_name(spec=False)) == len(specified_classes)
            self.specified_classes = specified_classes
        
        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()
        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        
        self._split()
        
        self.stack_num = stack_num
        
        self.img_channels = self.__getitem__(0)[0].shape[0]
        self.vis_idx = [300, 264, 179, 188, 42, 333, 48, 40, 147, 45,
                        17, 19, 24, 40, 119, 41, 60, 37, 68, 42,
                        29, 25, 253, 43, 38, 63, 176, 366, 27, 76,
                        13, 41, 57, 172, 45, 40, 55, 32, 21, 37,
                        67, 20, 164, 39, 37, 44, 70, 78, 30, 350,
                        41, 31, 166, 135, 32, 27, 42, 58, 29, 192,
                        18, 14, 34, 194, 27, 67, 190, 164, 274, 23]
    
    def _split(self):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        self.test_dataset = self
    
    def get_img_list(self, root, valid_rate):
        imgs = []
        labels = []
        
        roi_file = root / 'roi.json'
        assert roi_file.exists()
        with open(roi_file, 'r') as f:
            self.rois = json.load(f)
        
        self.case_indices = [0, ]
        cases = sorted([d for d in root.iterdir() if d.is_dir()])
        self.split_case = int(np.round(len(cases) * valid_rate))
        for i in range(len(cases)):
            case = cases[i]
            
            imaging_dir = case / 'imaging'
            segmentation_dir = case / 'segmentation'
            assert imaging_dir.exists() and segmentation_dir.exists()
            
            case_imgs = sorted(list(imaging_dir.glob('*.npy')))
            case_labels = sorted(list(segmentation_dir.glob('*.npy')))
            
            roi = self.rois[f'case_{i:05d}']
            d = 5
            min_z = max(0, roi['min_z'] - d)
            max_z = min(len(case_imgs) - 1, roi['max_z'] + d)
            
            imgs += case_imgs[min_z:max_z + 1]
            labels += case_labels[min_z:max_z + 1]
            
            assert len(imgs) == len(labels)
            self.case_indices.append(len(imgs))
            
            if case.stem[-3:] == f'{self.split_case:03}':
                split = len(imgs)
        
        self.indices = list(range(len(imgs)))
        self.train_indices = self.indices[split:]
        self.valid_indices = self.indices[:split]
        
        return imgs, labels
    
    def default_transform(self, data):
        # if data['image'].shape[1] != data['image'].shape[2] \
        #         and self.train_transform is None and self.valid_transform is None:
        #     data = kits19_transform()(data)
        
        image, label = data['image'], data['label']
        
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        label = label.astype(np.int64)
        
        idx = list(range(len(self.get_classes_name(spec=False))))
        masks = [np.where(label == i) for i in idx]
        
        spec_class_idx = []
        
        if self.specified_classes != [0, 1, 2]:
            for i in self.specified_classes:
                if i not in spec_class_idx:
                    spec_class_idx.append(i)
            
            for mask, spec_class in zip(masks, self.specified_classes):
                label[mask] = spec_class_idx.index(spec_class)
        
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return {'image': image, 'label': label}
    
    def get_colormap(self, spec=True):
        cmap = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
        cmap = np.array(cmap, dtype=np.int)
        
        if not spec:
            return cmap
        else:
            spec_cmap = []
            for i in cmap[self.specified_classes]:
                if len(spec_cmap) == 0:
                    spec_cmap.append(i)
                else:
                    duplicate = False
                    for j in spec_cmap:
                        duplicate = duplicate or (i == j).all()
                    if not duplicate:
                        spec_cmap.append(i)
            return np.array(spec_cmap)
    
    def get_classes_name(self, spec=True):
        classes_name = np.array(['background', 'kidney', 'tumor'])
        
        if not spec:
            return classes_name
        else:
            spec_classes_name = []
            for i in classes_name[self.specified_classes]:
                if i not in spec_classes_name:
                    spec_classes_name.append(i)
            return spec_classes_name
    
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
    
    def __getitem__(self, idx):
        for i in range(len(self.case_indices) - 1):
            if self.case_indices[i] <= idx < self.case_indices[i + 1]:
                case_i = i
                break
        
        roi = self.rois[f'case_{case_i:05d}']
        imgs = []
        for i in range(idx - self.stack_num // 2, idx + self.stack_num // 2 + 1):
            if i < self.case_indices[case_i]:
                i = self.case_indices[case_i]
            elif i >= self.case_indices[case_i + 1]:
                i = self.case_indices[case_i + 1] - 1
            img_path = self.imgs[i]
            img = np.load(str(img_path))
            imgs.append(img)
        
        img = np.stack(imgs, axis=2)
        img = img[roi['min_y']:roi['max_y'], roi['min_x']:roi['max_x'], :]
        
        label_path = self.labels[idx]
        label = np.load(str(label_path))
        label = label[roi['min_y']:roi['max_y'], roi['min_x']:roi['max_x']]
        
        data = {'image': img, 'label': label}
        if idx in self.train_indices and self.train_transform is not None:
            data = self.train_transform(data)
        elif idx in self.valid_indices and self.valid_transform is not None:
            data = self.valid_transform(data)
        
        data = self.default_transform(data)
        
        img = data['image']
        label = data['label']
        
        return img, label, idx
    
    def __len__(self):
        return self.dataset_size


class kits19_roi_volume(data.Dataset):
    def __init__(self, root, slice_num=1, valid_rate=0.3,
                 train_transform=None, valid_transform=None, specified_classes=None):
        self.root = Path(root)
        self.slice_num = slice_num
        
        self.imgs, self.labels = self.get_img_list(self.root, valid_rate)
        
        if specified_classes is None:
            self.specified_classes = [0, 1, 2]
        else:
            assert len(self.get_classes_name(spec=False)) == len(specified_classes)
            self.specified_classes = specified_classes
        
        self.dataset_size = len(self.imgs)
        self.classes_name = self.get_classes_name()
        self.num_classes = len(self.classes_name)
        self.cmap = self.get_colormap()
        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        
        self._split()
        
        self.img_channels = self.__getitem__(0)[0].shape[0]
        self.vis_idx = [300, 264, 179, 188, 42, 333, 48, 40, 147, 45,
                        17, 19, 24, 40, 119, 41, 60, 37, 68, 42,
                        29, 25, 253, 43, 38, 63, 176, 366, 27, 76,
                        13, 41, 57, 172, 45, 40, 55, 32, 21, 37,
                        67, 20, 164, 39, 37, 44, 70, 78, 30, 350,
                        41, 31, 166, 135, 32, 27, 42, 58, 29, 192,
                        18, 14, 34, 194, 27, 67, 190, 164, 274, 23]
    
    def _split(self):
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)
        self.test_dataset = self
    
    def get_img_list(self, root, valid_rate):
        imgs = []
        labels = []
        
        roi_file = root / 'roi.json'
        assert roi_file.exists()
        with open(roi_file, 'r') as f:
            self.rois = json.load(f)
        
        self.case_indices = [0, ]
        cases = sorted([d for d in root.iterdir() if d.is_dir()])
        self.split_case = int(np.round(len(cases) * valid_rate))
        for i in range(len(cases)):
            case = cases[i]
            
            imaging_dir = case / 'imaging'
            segmentation_dir = case / 'segmentation'
            assert imaging_dir.exists() and segmentation_dir.exists()
            
            case_imgs = sorted(list(imaging_dir.glob('*.npy')))
            case_labels = sorted(list(segmentation_dir.glob('*.npy')))
            
            roi = self.rois[f'case_{i:05d}']
            d = 5
            min_z = max(0, roi['min_z'] - d)
            max_z = min(len(case_imgs) - 1, roi['max_z'] + d)
            
            case_imgs = case_imgs[min_z:max_z + 1]
            case_labels = case_labels[min_z:max_z + 1]
            
            img_volumes = []
            label_volumes = []
            for i in range(0, len(case_imgs), self.slice_num):
                end = min(i + self.slice_num, len(case_imgs))
                img_volume = case_imgs[i:end]
                img_volumes.append(img_volume)
                label_volume = case_labels[i:end]
                label_volumes.append(label_volume)
            
            imgs += img_volumes
            labels += label_volumes
            
            assert len(imgs) == len(labels)
            self.case_indices.append(len(imgs))
            
            if case.stem[-3:] == f'{self.split_case:03}':
                split = len(imgs)
        
        self.indices = list(range(len(imgs)))
        self.train_indices = self.indices[split:]
        self.valid_indices = self.indices[:split]
        
        return imgs, labels
    
    def default_transform(self, data):
        # if data['image'].shape[1] != data['image'].shape[2] \
        #         and self.train_transform is None and self.valid_transform is None:
        #     data = kits19_transform()(data)
        
        image, label = data['image'], data['label']
        
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        label = label.astype(np.int64)
        
        idx = list(range(len(self.get_classes_name(spec=False))))
        masks = [np.where(label == i) for i in idx]
        
        spec_class_idx = []
        
        if self.specified_classes != [0, 1, 2]:
            for i in self.specified_classes:
                if i not in spec_class_idx:
                    spec_class_idx.append(i)
            
            for mask, spec_class in zip(masks, self.specified_classes):
                label[mask] = spec_class_idx.index(spec_class)
        
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return {'image': image, 'label': label}
    
    def get_colormap(self, spec=True):
        cmap = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
        cmap = np.array(cmap, dtype=np.int)
        
        if not spec:
            return cmap
        else:
            spec_cmap = []
            for i in cmap[self.specified_classes]:
                if len(spec_cmap) == 0:
                    spec_cmap.append(i)
                else:
                    duplicate = False
                    for j in spec_cmap:
                        duplicate = duplicate or (i == j).all()
                    if not duplicate:
                        spec_cmap.append(i)
            return np.array(spec_cmap)
    
    def get_classes_name(self, spec=True):
        classes_name = np.array(['background', 'kidney', 'tumor'])
        
        if not spec:
            return classes_name
        else:
            spec_classes_name = []
            for i in classes_name[self.specified_classes]:
                if i not in spec_classes_name:
                    spec_classes_name.append(i)
            return spec_classes_name
    
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
    
    def __getitem__(self, idx):
        for i in range(len(self.case_indices) - 1):
            if self.case_indices[i] <= idx < self.case_indices[i + 1]:
                case_i = i
                break
        
        roi = self.rois[f'case_{case_i:05d}']
        
        imgs = []
        labels = []
        for img_path, label_path in zip(self.imgs[idx], self.labels[idx]):
            img = np.load(str(img_path))
            img = img[roi['min_y']:roi['max_y'], roi['min_x']:roi['max_x']]
            imgs.append(img)
            
            label = np.load(str(label_path))
            label = label[roi['min_y']:roi['max_y'], roi['min_x']:roi['max_x']]
            labels.append(label)
        
        if len(imgs) != self.slice_num:
            num = self.slice_num - len(imgs)
            for _ in range(num):
                imgs.append(np.zeros((img.shape[0], img.shape[1])))
                labels.append(np.zeros((label.shape[0], label.shape[1])))
        
        transform_imgs = []
        transform_labels = []
        for img, label in zip(imgs, labels):
            data = {'image': img, 'label': label}
            if idx in self.train_indices and self.train_transform is not None:
                data = self.train_transform(data)
            elif idx in self.valid_indices and self.valid_transform is not None:
                data = self.valid_transform(data)
            data = self.default_transform(data)
            img = data['image']
            label = data['label']
            transform_imgs.append(img)
            transform_labels.append(label)
        
        img = np.stack(transform_imgs, axis=-1)
        label = np.stack(transform_labels, axis=-1)
        
        return img, label, idx
    
    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    import os
    
    root = os.path.expanduser('../data')
    dataset = kits19_roi_volume(root=root, slice_num=12, valid_rate=0.3,
                                train_transform=None,
                                valid_transform=None,
                                specified_classes=[0, 1, 2])
    
    from torch.utils.data import DataLoader, SequentialSampler
    from utils.vis import imshow
    
    subset = dataset.train_dataset
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=20, sampler=sampler)
    
    for batch_idx, (img, label, idx) in enumerate(data_loader):
        pass

import cv2
import numpy as np
from albumentations import Compose as Compose_alb
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    CenterCrop,
    GridDistortion,
    RandomCrop,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    RandomScale,
    Resize,
    OpticalDistortion,
    ElasticTransform
)


def to_numpy(data):
    image, label = data['image'], data['label']
    image = np.array(image)
    label = np.array(label)
    return {'image': image, 'label': label}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class RandomCropAndFlip:
    def __init__(self, output_size, type='train'):
        if isinstance(output_size, (tuple, list)):
            self.output_size = output_size  # (h, w)
        else:
            self.output_size = (output_size, output_size)

        self.type = type

    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']

        if self.type == 'train':
            aug = Compose_alb([
                PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                OneOf([
                    RandomCrop(height=self.output_size[0], width=self.output_size[1], p=1),
                    CenterCrop(height=self.output_size[0], width=self.output_size[1], p=1)
                ], p=1),
                HorizontalFlip(p=0.5),
            ])
        elif self.type == 'valid':
            aug = Compose_alb([
                PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                CenterCrop(height=self.output_size[0], width=self.output_size[1], p=1)
            ])

        data = aug(image=img, mask=label)
        img, label = data['image'], data['mask']

        data = {'image': img, 'label': label}
        return data


class PadAndResize:
    def __init__(self, output_size, type='train'):
        if isinstance(output_size, (tuple, list)):
            self.output_size = output_size  # (h, w)
        else:
            self.output_size = (output_size, output_size)

        self.type = type

    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']

        aug = Compose_alb([
            PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                        border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            Resize(height=self.output_size[0], width=self.output_size[1])
        ])

        data = aug(image=img, mask=label)
        img, label = data['image'], data['mask']

        data = {'image': img, 'label': label}
        return data


class RandomScaleCrop:
    def __init__(self, output_size, scale_range=0.1, type='train'):
        if isinstance(output_size, (tuple, list)):
            self.output_size = output_size  # (h, w)
        else:
            self.output_size = (output_size, output_size)

        self.scale_range = scale_range
        self.type = type

    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']

        img_size = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
        crop_size = self.output_size[0] if self.output_size[0] < self.output_size[1] else self.output_size[1]
        scale = crop_size / img_size - 1
        if scale < 0:
            scale_limit = (scale - self.scale_range, scale + self.scale_range)
        else:
            scale_limit = (-self.scale_range, scale + self.scale_range)

        if self.type == 'train':
            aug = Compose_alb([
                RandomScale(scale_limit=scale_limit, p=1),
                PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                OneOf([
                    RandomCrop(height=self.output_size[0], width=self.output_size[1], p=1),
                    CenterCrop(height=self.output_size[0], width=self.output_size[1], p=1)
                ], p=1),
            ])
        elif self.type == 'valid':
            aug = Compose_alb([
                PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                CenterCrop(height=self.output_size[0], width=self.output_size[1], p=1)
            ])

        data = aug(image=img, mask=label)
        img, label = data['image'], data['mask']

        data = {'image': img, 'label': label}
        return data


class MedicalTransform:
    def __init__(self, type):
        self.type = type

    def __call__(self, data):
        data = to_numpy(data)

        img, label = data['image'], data['label']

        if self.type == 'train':
            aug = Compose_alb([
                HorizontalFlip(p=0.5),
                OneOf([
                    GridDistortion(p=1, border_mode=cv2.BORDER_CONSTANT),
                    # OpticalDistortion(p=1, distort_limit=1, shift_limit=10)
                ], p=0.5),
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.5)
            ])
            data = aug(image=img, mask=label)
            img, label = data['image'], data['mask']

        data = {'image': img, 'label': label}
        return data


class MedicalTransform3D:
    def __init__(self, type):
        self.type = type

    def __call__(self, data):
        data = to_numpy(data)

        img, label = data['image'], data['label']

        if self.type == 'train':
            aug = Compose_alb([
                HorizontalFlip(p=0.5),
                OneOf([
                    GridDistortion(p=1, border_mode=cv2.BORDER_CONSTANT),
                    # OpticalDistortion(p=1, distort_limit=1, shift_limit=10, border_mode=cv2.BORDER_CONSTANT),
                    # ElasticTransform(p=1, border_mode=cv2.BORDER_CONSTANT)

                ], p=0.5),
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.5)
            ])

            keys = {}
            targets = {}
            for i in range(1, img.shape[2]):
                keys.update({f'image{i}': 'image'})
                keys.update({f'mask{i}': 'mask'})
                targets.update({f'image{i}': img[:, :, i]})
                targets.update({f'mask{i}': label[:, :, i]})
            aug.add_targets(keys)

            targets.update({'image': img[:, :, 0]})
            targets.update({'mask': label[:, :, 0]})

            data = aug(**targets)
            imgs = [data['image']]
            labels = [data['mask']]

            for i in range(1, img.shape[2]):
                imgs.append(data[f'image{i}'])
                labels.append(data[f'mask{i}'])

            img = np.stack(imgs, axis=-1)
            label = np.stack(labels, axis=-1)

        data = {'image': img, 'label': label}
        return data

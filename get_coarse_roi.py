import json
from argparse import ArgumentParser

import cv2
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader, SequentialSampler

import utils.checkpoint as cp
from dataset import kits19
from network import ResUNet
from utils.func import *
from utils.vis import imshow


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='resume checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    cp_file = Path(args.resume).expanduser()
    assert cp_file.is_file()

    BATCH_SIZE = 2
    num_workers = 0
    visualize_iter_interval = 20

    data_path = Path('data')
    dataset = kits19(data_path, stack_num=5, valid_rate=0.3,
                     train_transform=None,
                     valid_transform=None,
                     specified_classes=[0, 1, 1])

    net = ResUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=64)

    gpu_ids = [0]
    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()

    net, _, _ = cp.load_params(net, root=str(cp_file))

    net.eval()
    torch.set_grad_enabled(False)

    subset = dataset.test_dataset
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=BATCH_SIZE, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)

    case_i = 0
    min_x = min_y = min_z = 10000
    max_x = max_y = max_z = -1

    rois = {}
    roi_file = data_path / 'roi.json'

    tbar = tqdm(data_loader, desc='eval', ascii=True, dynamic_ncols=True)
    for batch_idx, (imgs, labels, idx) in enumerate(tbar):
        imgs = imgs.cuda()
        outputs = net(imgs).argmax(dim=1)

        np_outputs = outputs.cpu().detach().numpy()
        np_labels = labels.cpu().detach().numpy()
        idx = idx.numpy()

        for i, output in zip(idx, np_outputs):
            if output.max() != 0:
                i = int(i) - dataset.case_indices[case_i]
                rect = cv2.boundingRect(output.astype(np.uint8))
                min_x = min(min_x, rect[0])
                min_y = min(min_y, rect[1])
                min_z = min(min_z, i)
                max_x = max(max_x, rect[0] + rect[2])
                max_y = max(max_y, rect[1] + rect[3])
                max_z = max(max_z, i)

            if i >= dataset.case_indices[case_i + 1] - 1:
                roi = {'min_x': min_x, 'min_y': min_y, 'min_z': min_z, 'max_x': max_x, 'max_y': max_y, 'max_z': max_z}
                rois.update({f'case_{case_i:05d}': roi})

                with open(roi_file, 'w') as f:
                    json.dump(rois, f, sort_keys=True, indent=4, separators=(',', ': '))

                min_x = min_y = min_z = 10000
                max_x = max_y = max_z = -1
                case_i += 1

        if visualize_iter_interval > 0 and batch_idx % visualize_iter_interval == 0:
            vis_imgs, vis_labels, vis_outputs = dataset.vis_transform(imgs, labels, outputs)
            cv2.line(vis_imgs[0][2], (min_x, min_y), (min_x, max_y), 0.1, 1)
            cv2.line(vis_imgs[0][2], (min_x, min_y), (max_x, min_y), 0.1, 1)
            cv2.line(vis_imgs[0][2], (min_x, max_y), (max_x, max_y), 0.1, 1)
            cv2.line(vis_imgs[0][2], (max_x, min_y), (max_x, max_y), 0.1, 1)
            imshow(title='Valid', imgs=(vis_imgs[0][2], vis_labels[0], vis_outputs[0]), shape=(1, 3),
                   subtitle=('image', 'label', 'predict'))

import multiprocessing as mp

import click
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS19_vol
from dataset.transform import Compose, PadAndResize, MedicalTransform3D
from network import DenseUNet2D, HybridNet
from utils.vis import Plot
from utils.vis_boundary import vis_boundary


@click.command()
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=1, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('--data', 'data_path', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('-o', '--output', 'output_path', help='output image path',
              type=click.Path(dir_okay=True, resolve_path=True), default='out', show_default=True)
@click.option('--du2d', 'du2d_path', help='DenseUNet2D checkpoint path',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), required=True)
@click.option('--hybd', 'hybrid_path', help='HybridNet checkpoint path',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), required=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
def main(batch_size, num_gpu, data_path, output_path, du2d_path, hybrid_path, num_workers):
    # prepare
    data_path = Path(data_path)
    du2d_path = Path(du2d_path)
    output_path = Path(output_path)

    valid_transform = Compose([
        PadAndResize(output_size=224, type='valid'),
        MedicalTransform3D(type='valid')
    ])
    dataset = KiTS19_vol(data_path, slice_num=12, valid_rate=0.3,
                         train_transform=valid_transform,
                         valid_transform=valid_transform,
                         spec_classes=[0, 1, 2])

    net = HybridNet(in_ch=4, out_ch=dataset.num_classes)
    net, _, _ = cp.load_params(net, root=str(hybrid_path))

    dense_unet_2d = DenseUNet2D(out_ch=dataset.num_classes)
    dense_unet_2d, _, _ = cp.load_params(dense_unet_2d, root=str(du2d_path))

    gpu_ids = [i for i in range(num_gpu)]

    torch.cuda.empty_cache()

    # to GPU device
    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()
    dense_unet_2d = torch.nn.DataParallel(dense_unet_2d, device_ids=gpu_ids).cuda()

    net.eval()
    dense_unet_2d.eval()
    torch.set_grad_enabled(False)

    evaluation(net, dense_unet_2d, dataset, batch_size, num_workers, output_path, type='train')
    evaluation(net, dense_unet_2d, dataset, batch_size, num_workers, output_path, type='valid')


def evaluation(net, dense_unet_2d, dataset, batch_size, num_workers, output_path, type):
    type = type.lower()
    if type == 'train':
        subset = dataset.train_dataset
        case = dataset.case_indices[dataset.split_case:]
    elif type == 'valid':
        subset = dataset.valid_dataset
        case = dataset.case_indices[:dataset.split_case + 1]

    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)

    vol_case_i = 0
    with tqdm(total=len(case) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, (imgs, labels, idx) in enumerate(data_loader):
            imgs = imgs.cuda()

            slice_num = imgs.shape[4]
            stack_imgs = []
            for i in range(-1, slice_num - 1):
                stack_img = []
                for j in range(3):
                    k = i + j
                    k = max(k, 0)
                    k = min(k, slice_num - 1)
                    stack_img.append(imgs[:, :, :, :, k])

                stack_img = torch.cat(stack_img, dim=1)
                stack_imgs.append(stack_img)

            feat_2d_list = []
            outputs_2d_list = []
            for stack_img in stack_imgs:
                feat_2d, outputs_2d = dense_unet_2d(stack_img)
                feat_2d_list.append(feat_2d.detach())
                outputs_2d_list.append(outputs_2d.detach())

            outputs_2d = torch.stack(outputs_2d_list, dim=-1)
            feat_2d = torch.stack(feat_2d_list, dim=-1)
            input_concat = torch.cat((imgs, outputs_2d), dim=1)

            feat_3d, cls_3d, outputs = net(input_concat, feat_2d)

            outputs = outputs.argmax(dim=1)

            idx = idx.numpy()
            np_imgs = imgs.cpu().detach().numpy()
            np_labels = labels.cpu().detach().numpy()
            np_outputs = outputs.cpu().detach().numpy()

            while vol_case_i < len(case) - 1 and idx[-1] >= case[vol_case_i + 1] - 1:
                vol_case_i += 1
                pbar.update(1)

            # mp.freeze_support()

            names_list = [dataset.idx_to_name(idx_) for idx_ in idx]
            output_path_list = [output_path] * len(names_list)
            type_list = [type] * len(names_list)

            pool = mp.Pool()
            pool.map(output_img, zip(names_list, type_list, np_imgs, np_labels, np_outputs, output_path_list))
            pool.close()
            pool.join()


def output_img(data):
    names, type, imgs, labels, outputs, output_path = data
    plot = Plot(type, (1, 3), ('image', 'cls0', 'cls1'))
    for i in range(len(names)):
        output_file = output_path / f'{names[i]}.png'
        if not output_file.parent.exists():
            try:
                output_file.parent.mkdir(parents=True)
            except FileExistsError:
                pass

        vis_img = vis_boundary(imgs[0, :, :, i], labels[:, :, i], outputs[:, :, i], 3)
        plot.set_img((imgs[0, :, :, i], vis_img[0], vis_img[1]))
        # plot.show()
        plot.save(str(output_file), dpi=200)


if __name__ == '__main__':
    main()

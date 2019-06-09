import sys

import click
import numpy as np
import torch
import torch.nn as nn
from pathlib2 import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS19_roi
from dataset.transform import Compose, MedicalTransform2
from network import DenseUNet2D
from utils.metrics import Evaluator
from utils.vis import imshow


@click.command()
@click.option('-e', '--epoch', 'epoch_num', help='Number of training epoch', type=int, default=1, show_default=True)
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=1, show_default=True)
@click.option('-l', '--lr', help='Learning rate', type=float, default=0.0001, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('--data', 'data_path', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('--log', 'log_path', help='Checkpoint and log file save path',
              type=click.Path(dir_okay=True, resolve_path=True),
              default='runs', show_default=True)
@click.option('-r', '--resume', help='Resume checkpoint file to continue training',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), default=None)
@click.option('--eval_intvl', help='Number of epoch interval of evaluation. '
                                   'No evaluation when set to 0',
              type=int, default=1, show_default=True)
@click.option('--cp_intvl', help='Number of epoch interval of checkpoint save. '
                                 'No checkpoint save when set to 0',
              type=int, default=1, show_default=True)
@click.option('--vis_intvl', help='Number of iteration interval of display visualize image. '
                                  'No display when set to 0',
              type=int, default=20, show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
def main(epoch_num, batch_size, lr, num_gpu, data_path, log_path, resume, eval_intvl, cp_intvl, vis_intvl, num_workers):
    # prepare
    data_path = Path(data_path)
    log_path = Path(log_path)
    cp_path = log_path / 'checkpoint'

    if not resume and log_path.exists() and len(list(log_path.glob('*'))) > 0:
        print(f'log path "{str(log_path)}" has old file', file=sys.stderr)
        sys.exit(-1)
    if not cp_path.exists():
        cp_path.mkdir(parents=True)

    train_transform = Compose([
        MedicalTransform2(output_size=512, type='train')
    ])
    valid_transform = Compose([
        MedicalTransform2(output_size=512, type='valid')
    ])
    dataset = KiTS19_roi(data_path, stack_num=3, valid_rate=0.3,
                         train_transform=train_transform,
                         valid_transform=valid_transform,
                         spec_classes=[0, 1, 2])

    net = DenseUNet2D(out_ch=dataset.num_classes)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    start_epoch = 0
    if resume:
        cp_file = Path(resume)
        net, optimizer, start_epoch = cp.load_params(net, optimizer, root=str(cp_file))

    # weights = np.array([0.2, 1.2, 2.2], dtype=np.float32)
    # weights = torch.from_numpy(weights)
    weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )

    logger = SummaryWriter(str(log_path))

    gpu_ids = [i for i in range(num_gpu)]

    print(f'{" Start training ":-^40s}\n')
    msg = f'Net: {net.__class__.__name__}\n' + \
          f'Dataset: {dataset.__class__.__name__}\n' + \
          f'Epochs: {epoch_num}\n' + \
          f'Learning rate: {optimizer.param_groups[0]["lr"]}\n' + \
          f'Batch size: {batch_size}\n' + \
          f'Device: cuda{str(gpu_ids)}\n'
    print(msg)

    torch.cuda.empty_cache()

    # to GPU device
    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()
    criterion = criterion.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # start training
    valid_score = 0.0
    best_score = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, epoch_num):
        epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
        print(f'{epoch_str:-^40s}')

        lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {lr}')

        net.train()
        torch.set_grad_enabled(True)
        try:
            loss = training(net, dataset, criterion, optimizer, scheduler, batch_size, num_workers, vis_intvl, logger, epoch)

            if eval_intvl > 0 and (epoch + 1) % eval_intvl == 0:
                net.eval()
                torch.set_grad_enabled(False)

                train_score = evaluation(net, dataset, batch_size, num_workers, vis_intvl, logger, epoch, type='train')
                valid_score = evaluation(net, dataset, batch_size, num_workers, vis_intvl, logger, epoch, type='valid')

                print(f'Train data score: {train_score:.5f}')
                print(f'Valid data score: {valid_score:.5f}')

        except KeyboardInterrupt:
            cp_file = cp_path / 'INTERRUPTED.pth'
            cp.save(epoch, net.module, optimizer, str(cp_file))
            return

        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            cp_file = cp_path / 'best.pth'
            cp.save(epoch, net.module, optimizer, str(cp_file))
            print('Update best acc!')
            logger.add_scalar('best epoch', best_epoch + 1, 0)
            logger.add_scalar('best score', best_score, 0)

        if (epoch + 1) % cp_intvl == 0:
            cp_file = cp_path / f'cp_{epoch + 1:03d}.pth'
            cp.save(epoch, net.module, optimizer, str(cp_file))

        print(f'Best epoch: {best_epoch + 1}')
        print(f'Best score: {best_score:.5f}')


def training(net, dataset, criterion, optimizer, scheduler, batch_size, num_workers, vis_intvl, logger, epoch):
    sampler = RandomSampler(dataset.train_dataset)

    train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)

    tbar = tqdm(train_loader, ascii=True, desc='train', dynamic_ncols=True)
    for batch_idx, (imgs, labels, idx) in enumerate(tbar):
        optimizer.zero_grad()

        imgs, labels = imgs.cuda(), labels.cuda()
        feat, outputs = net(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if vis_intvl > 0 and batch_idx % vis_intvl == 0:
            outputs = outputs.cpu().detach().numpy().argmax(axis=1)
            imgs, labels, outputs = dataset.vis_transform(imgs, labels, outputs)
            imshow(title='Train', imgs=(imgs[0][1], labels[0], outputs[0]), shape=(1, 3),
                   subtitle=('image', 'label', 'predict'))

        tbar.set_postfix(loss=f'{loss.item():.5f}')

    scheduler.step(loss.item())

    logger.add_scalar('loss', loss.item(), epoch)
    return loss.item()


def evaluation(net, dataset, batch_size, num_workers, vis_intvl, logger, epoch, type):
    type = type.lower()
    if type == 'train':
        subset = dataset.train_dataset
        case = dataset.case_indices[dataset.split_case:]
    elif type == 'valid':
        subset = dataset.valid_dataset
        case = dataset.case_indices[:dataset.split_case + 1]

    vol_case_i = 0
    vol_label = []
    vol_output = []

    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)

    evaluator = Evaluator(dataset.num_classes)

    with tqdm(total=len(case) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, (imgs, labels, idx) in enumerate(data_loader):
            imgs = imgs.cuda()
            feat, outputs = net(imgs)
            outputs = outputs.argmax(dim=1)

            np_labels = labels.cpu().detach().numpy()
            np_outputs = outputs.cpu().detach().numpy()
            idx = idx.numpy()

            vol_label.append(np_labels)
            vol_output.append(np_outputs)

            while vol_case_i < len(case) - 1 and idx[-1] >= case[vol_case_i + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_label = np.concatenate(vol_label, axis=0)

                vol_idx = case[vol_case_i + 1] - case[vol_case_i]
                evaluator.add(vol_output[:vol_idx], vol_label[:vol_idx])

                vol_output = [vol_output[vol_idx:]]
                vol_label = [vol_label[vol_idx:]]
                vol_case_i += 1
                pbar.update(1)

            if vis_intvl > 0 and batch_idx % vis_intvl == 0:
                imgs, labels, outputs = dataset.vis_transform(imgs, labels, outputs)
                imshow(title='Train', imgs=(imgs[0][1], labels[0], outputs[0]), shape=(1, 3),
                       subtitle=('image', 'label', 'predict'))

    acc = evaluator.eval()

    for k in sorted(list(acc.keys())):
        if k == 'dc_each_case': continue
        print(f'{type}/{k}: {acc[k]:.5f}')
        logger.add_scalar(f'{type}/{k}', acc[k], epoch)

    for i in range(len(acc['dc_each_case'])):
        dc_each_case = acc['dc_each_case'][i]
        for j in range(len(dc_each_case)):
            dc = dc_each_case[j]
            logger.add_scalar(f'{type}_each_case/{i:05d}/dc_{j}', dc, epoch)

    score = (acc['dc_per_case_1'] + acc['dc_per_case_2']) / 2
    logger.add_scalar(f'{type}/score', score, epoch)
    return score


if __name__ == '__main__':
    main()

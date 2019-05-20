from argparse import ArgumentParser

import torch
import torch.nn as nn
from pathlib2 import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import utils.checkpoint as cp
from dataset import kits19
from dataset.transform import Compose, RandomScaleCrop, MedicalTransform
from network import ResUNet
from utils.func import *
from utils.metrics import Evaluator
from utils.vis import imshow


class Trainer:
    def __init__(self, net, dataset, criterion, optimizer, scheduler, epoch_num, start_epoch, batch_size):
        self.net = net
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_num = epoch_num
        self.start_epoch = start_epoch
        self.batch_size = batch_size

        self.cuda = True
        self.gpu_ids = [0]
        self.eval_func = 'dc'
        self.visualize_iter_interval = 1
        self.checkpoint_epoch_interval = 10
        self.eval_epoch_interval = 10
        self.num_workers = 0
        self.checkpoint_dir = 'runs'

    def run(self):
        self.net = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids).cuda()
        self.criterion = self.criterion.cuda()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        valid_acc = 0.0
        best_acc = 0.0
        best_epoch = 0

        for epoch in range(self.start_epoch, self.epoch_num):
            epoch_str = ' Epoch {}/{} '.format(epoch + 1, self.epoch_num)
            print('{:-^40s}'.format(epoch_str))

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                break
            print('Learning rate: {}'.format(lr))

            # Training phase
            self.net.train()
            torch.set_grad_enabled(True)

            try:
                valid_acc = self.evaluation(epoch, 'valid')
                loss = self.training(epoch)

                if (epoch + 1) % self.eval_epoch_interval == 0:
                    self.net.eval()
                    torch.set_grad_enabled(False)

                    train_acc = self.evaluation(epoch, 'train')

                    valid_acc = self.evaluation(epoch, 'valid')

                    print('Train data {} acc:  {:.5f}'.format(self.eval_func, train_acc))
                    print('Valid data {} acc:  {:.5f}'.format(self.eval_func, valid_acc))

            except KeyboardInterrupt:
                cp_path = os.path.join(self.checkpoint_dir, 'INTERRUPTED.pth')
                cp.save(epoch, self.net.module, self.optimizer, cp_path)
                return

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch
                checkpoint_filename = 'best.pth'
                cp.save(epoch, self.net.module, self.optimizer, os.path.join(self.checkpoint_dir, checkpoint_filename))
                print('Update best acc!')

            if (epoch + 1) % self.checkpoint_epoch_interval == 0:
                checkpoint_filename = 'cp_{:03d}.pth'.format(epoch + 1)
                cp.save(epoch, self.net.module, self.optimizer, os.path.join(self.checkpoint_dir, checkpoint_filename))

            print(f'Best epoch: {best_epoch + 1}')
            print(f'Best acc: {best_acc:.5f}')

    def training(self, epoch):
        sampler = RandomSampler(self.dataset.train_dataset)

        train_loader = DataLoader(self.dataset.train_dataset, batch_size=self.batch_size, sampler=sampler,
                                  num_workers=self.num_workers, pin_memory=True)

        tbar = tqdm(train_loader, ascii=True, desc='train', dynamic_ncols=True)
        for batch_idx, (imgs, labels, idx) in enumerate(tbar):
            self.optimizer.zero_grad()

            if self.cuda:
                imgs, labels = imgs.cuda(), labels.cuda()

            outputs = self.net(imgs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if self.visualize_iter_interval > 0 and batch_idx % self.visualize_iter_interval == 0:
                outputs = outputs.cpu().detach().numpy().argmax(axis=1)
                imgs, labels, outputs = self.dataset.vis_transform(imgs, labels, outputs)
                imshow(title='Train', imgs=(imgs[0][2], labels[0], outputs[0]), shape=(1, 3),
                       subtitle=('image', 'label', 'predict'))

            tbar.set_postfix(loss='{:.5f}'.format(loss.item()))
        self.scheduler.step(loss.item())

        return loss.item()

    def evaluation(self, epoch, type):
        type = type.lower()
        if type == 'train':
            subset = self.dataset.train_dataset
            case = self.dataset.case_indices[self.dataset.split_case:]
        elif type == 'valid':
            subset = self.dataset.valid_dataset
            case = self.dataset.case_indices[:self.dataset.split_case + 1]

        vol_case_i = 0
        vol_label = []
        vol_output = []

        sampler = SequentialSampler(subset)
        data_loader = DataLoader(subset, batch_size=self.batch_size, sampler=sampler,
                                 num_workers=self.num_workers, pin_memory=True)

        evaluator = Evaluator(self.dataset.num_classes)

        with tqdm(total=len(case), ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
            for batch_idx, (imgs, labels, idx) in enumerate(data_loader):
                if self.cuda:
                    imgs = imgs.cuda()
                outputs = self.net(imgs).argmax(dim=1)

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

        acc = evaluator.eval(self.eval_func)
        evaluator.print_acc()
        return acc


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='resume checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    EPOCH = 100
    BATCH_SIZE = 32
    LR = 0.0001

    data_path = Path('data')

    train_transform = Compose([
        RandomScaleCrop(output_size=512, scale_range=0.2, type='train'),
        MedicalTransform(type='train')
    ])
    valid_transform = Compose([
        RandomScaleCrop(output_size=512, scale_range=0.2, type='valid'),
        MedicalTransform(type='valid')
    ])

    dataset = kits19(data_path, stack_num=5, valid_rate=0.3,
                     train_transform=train_transform,
                     valid_transform=valid_transform,
                     specified_classes=[0, 1, 1])

    net = ResUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes, base_ch=64)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    start_epoch = 0
    if args.resume:
        cp_file = Path(args.resume)
        net, optimizer, start_epoch = cp.load_params(net, optimizer, root=cp_file)

    # weights = np.array([0.2, 1.2, 2.2], dtype=np.float32)
    # weights = torch.from_numpy(weights)
    criterion = nn.CrossEntropyLoss(weight=None)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )

    torch.cuda.empty_cache()
    trainer = Trainer(net, dataset, criterion, optimizer, scheduler, epoch_num=EPOCH, start_epoch=start_epoch,
                      batch_size=BATCH_SIZE)
    trainer.run()

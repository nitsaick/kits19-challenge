import io
import os
from contextlib import redirect_stdout

import numpy as np
from torchsummary import summary
from tqdm import tqdm


def recursive_glob(root='.', suffix=''):
    """Performs recursive glob with given suffix and root
        :param root is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(root)
            for filename in filenames if filename.endswith(suffix)]


def count_params(net):
    return sum([param.nelement() for param in net.parameters()])


def calc_class_weigth(dataset):
    num_classes = dataset.num_classes
    dataloader, _, _ = dataset.get_dataloader()
    class_count = np.zeros((num_classes,))

    print('Calculating classes weights')
    tqdm_batch = tqdm(dataloader, ascii=True)
    for _, label in tqdm_batch:
        label = label.numpy()
        mask = (label >= 0) & (label < num_classes)
        labels = label[mask].astype(np.uint8)
        count = np.bincount(labels, minlength=num_classes)
        class_count += count
    # class_count = class_count[1:]
    total = np.sum(class_count)
    class_weights = []
    for count in class_count:
        weight = total / count
        class_weights.append(weight)
    class_weights /= np.mean(class_weights)
    # class_weights = [1, *class_weights]
    class_weights = np.array(class_weights)

    path = os.path.join(dataset.root, 'class_weights.npy')
    np.save(path, class_weights)

    path = os.path.join(dataset.root, 'class_count.npy')
    np.save(path, class_count)

    return class_weights


def net_summary(net, dataset, device='cpu'):
    with io.StringIO() as buf, redirect_stdout(buf):
        if device == 'cpu':
            summary(net, dataset.__getitem__(0)[0].shape, device=device)
        elif device == 'cuda':
            summary(net.cuda(), dataset.__getitem__(0)[0].shape, device=device)
        net_summary_text = buf.getvalue()

        return net_summary_text

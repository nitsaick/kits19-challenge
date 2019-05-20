import matplotlib.pyplot as plt
import torch


def numpy_to_plt(img):
    return img.transpose((1, 2, 0))


def imshow(title, imgs, shape=None, subtitle=None, cmap=None, transpose=False):
    if type(imgs) is tuple:
        num = len(imgs)
        if shape is not None:
            assert shape[0] * shape[1] == num
        else:
            shape = (1, num)
        
        if type(subtitle) is not tuple:
            subtitle = (subtitle,) * num
        else:
            assert len(subtitle) == num
        
        if type(cmap) is not tuple:
            cmap = (cmap,) * num
        else:
            assert len(cmap) == num
        
        fig = plt.figure(num=title, figsize=(shape[1] * 3, shape[0] * 3 + 0.5))
        fig.clf()
        fig.suptitle(title)
        
        fig.subplots(shape[0], shape[1], sharex=True, sharey=True)
        axes = fig.get_axes()
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                idx = i * shape[1] + j
                axes[idx].set_title(subtitle[idx])
                
                cm = cmap[idx]
                img = imgs[idx]
                if cmap[idx] is None and len(img.shape) == 3:
                    if img.shape[0] == 1 or len(img.shape) == 2:
                        cm = 'gray'
                        if len(img.shape) == 3 and img.shape[0] == 1:
                            img = img.reshape((img.shape[1], img.shape[2]))
                    elif img.shape[0] == 3:
                        img = numpy_to_plt(img)
                axes[idx].imshow(img, cm)
    
    else:
        if transpose:
            imgs = numpy_to_plt(imgs)
        plt.figure(num=title)
        plt.suptitle(title)
        plt.title(subtitle)
        plt.imshow(imgs, cmap)
    
    plt.ion()
    plt.show()
    plt.pause(0.001)


if __name__ == '__main__':
    img = torch.rand(50, 50)
    imshow('Test', (img, img, img, img), (2, 2), ('a', 'b', 'c', 'd'))

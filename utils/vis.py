import matplotlib.pyplot as plt
import torch


def numpy_to_plt(img):
    return img.transpose((1, 2, 0))


class Plot:
    def __init__(self, title=None, shape=None, subtitle=None, cmap=None):
        assert len(shape) == 1 or len(shape) == 2
        if len(shape) == 1:
            shape = (1, shape)
        num = shape[0] * shape[1]
        
        if type(subtitle) is not tuple or len(subtitle) == 1:
            subtitle = (subtitle,) * num
        assert len(subtitle) == num
        
        if type(cmap) is not tuple or len(cmap) == 1:
            cmap = (cmap,) * num
        assert len(cmap) == num
        
        fig = plt.figure(num=title, figsize=(shape[1] * 3, shape[0] * 3 + 0.5))
        fig.clf()
        if title is not None:
            fig.suptitle(title)
        
        fig.subplots(shape[0], shape[1], sharex=True, sharey=True)
        axes = fig.get_axes()
        
        self.shape = shape
        self.subtitle = subtitle
        self.cmap = cmap
        self.fig = fig
        self.axes = axes
    
    def set_img(self, imgs):
        if type(imgs) is not tuple:
            imgs = (imgs,)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                idx = i * self.shape[1] + j
                
                cm = self.cmap[idx]
                img = imgs[idx]
                if self.cmap[idx] is None and len(img.shape) == 3:
                    if img.shape[0] == 1 or len(img.shape) == 2:
                        cm = 'gray'
                        if len(img.shape) == 3 and img.shape[0] == 1:
                            img = img.reshape((img.shape[1], img.shape[2]))
                    elif img.shape[0] == 3:
                        img = numpy_to_plt(img)
                
                self.axes[idx].clear()
                self.axes[idx].set_title(self.subtitle[idx])
                self.axes[idx].imshow(img, cm)
    
    def show(self, pause=0.0001):
        plt.ion()
        plt.show()
        plt.pause(pause)
    
    def save(self, filename, dpi=100):
        self.fig.savefig(filename, dpi=dpi)


def imshow(title, imgs, shape=None, subtitle=None, cmap=None, transpose=False, pause=0.001, pltshow=True):
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
    
    if pltshow:
        plt.ion()
        plt.show()
        plt.pause(pause)
    
    return plt.gcf()


if __name__ == '__main__':
    img = torch.rand(50, 50)
    # fig = imshow('Test', (img, img, img, img), (2, 2), ('a', 'b', 'c', 'd'))
    # fig.savefig('test.png')
    
    plot = Plot('Test', (2, 2), ('a', 'b', 'c', 'd'))
    plot.set_img((img, img, img, img))
    plot.show(0.0001)
    
    ...

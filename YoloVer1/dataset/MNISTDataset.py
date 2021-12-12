from typing import Callable, Optional

import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

from YoloVer1.dataset.PlotMNISTImage import plot_mnist_image
from YoloVer1.grids.YoloGrids import YoloGrids


class GenRandMNISTImage(object):

    def __init__(self, img_size: tuple = (448, 448, 3), rand_range: tuple = (1.5, 5.), obj_size: tuple = (32, 32)):
        self.new_img_size = img_size
        self.rand_range = rand_range
        self.obj_size = obj_size

    def __call__(self, img: np.ndarray):
        # randomly scale the imageq
        rand_scale = np.random.uniform(self.rand_range[0], self.rand_range[1])
        rand_w = int(self.obj_size[0] * rand_scale)
        rand_h = int(self.obj_size[1] * rand_scale)

        # randomly object coordinates
        shift_x = np.random.randint(0, self.new_img_size[0] - rand_w)
        shift_y = np.random.randint(0, self.new_img_size[1] - rand_h)

        # convert gray image to color image
        img = np.expand_dims(img, axis=2)
        img = np.concatenate([img, img, img], axis=2)

        # resize the image
        img = cv2.resize(img, (rand_w, rand_h), interpolation=cv2.INTER_CUBIC)

        # create a new image with the given size
        blank_image = np.zeros(self.new_img_size, np.uint8)

        # we now need to put the picture in a random position of the new image
        blank_image[shift_y: shift_y + rand_h, shift_x: shift_x + rand_w] = img

        # swap the dimensions of the image
        blank_image = np.transpose(blank_image, (2, 0, 1))

        # 返回新的图片以及 bounding box 坐标信息
        return blank_image, (shift_x, shift_y, shift_x + rand_w, shift_y + rand_h)


class MNISTDataset(datasets.MNIST):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 grids_system: Optional[Callable] = None,
                 rand_mnist: Optional[Callable] = None):

        # call super constructor
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # 生成随机MNIST图像
        self.rand_mnist = rand_mnist

        # create the net grids
        self.grids_system = grids_system

    def __getitem__(self, index):
        # get the image and the label
        img, target = self.data[index], int(self.targets[index])

        # bounding box
        bbox = None

        # 是否需要生成新的MNIST图像
        if self.rand_mnist is not None:
            img, bbox = self.rand_mnist(img)

        # 图片数据是否需要转化
        if self.transform is not None:
            img = self.transform(img)

        # 标签信息是否需要转化
        if self.target_transform is not None:
            target = self.target_transform(target)

        elif self.grids_system is not None and bbox is not None:
            target = self.grids_system(target, bbox)

        return img, target

    def __len__(self):
        return super().__len__()


def test():
    # create yolo grids
    yolo_grids = YoloGrids()

    # resize the output MNIST image
    rand_mnist = GenRandMNISTImage()

    # create the MNIST dataset
    dataset = MNISTDataset(root='../data/MNIST', train=True, download=False,
                           rand_mnist=rand_mnist, grids_system=yolo_grids)

    # create data loader
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    # iterate over the data
    for i, (img, target) in enumerate(data_loader):
        # show images
        plot_mnist_image(images=img, marks=target)

        if i == 10:
            break
        

if __name__ == '__main__':
    test()

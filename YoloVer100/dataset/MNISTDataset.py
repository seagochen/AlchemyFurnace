from typing import Callable, Optional

import cv2
from torch.utils.data import DataLoader
from torchvision import datasets

from YoloVer100.dataset.PlotMNISTImage import plot_mnist_image
from YoloVer100.grids.YoloGrids import YoloGrids
from YoloVer100.tools.Normalizer import *


class GenerateRandMNIST(object):

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
                 download: bool = False,
                 norm_data: Optional[Callable] = None,
                 grids_system: Optional[Callable] = None,
                 rand_mnist: Optional[Callable] = None):

        # call super constructor
        super().__init__(root=root, train=train, download=download)

        # generate rand position and scaled MNIST image
        self.rand_mnist = rand_mnist

        # use yolo grid system
        self.grids_system = grids_system

        # use normalize function
        self.norm_data = norm_data

    def __getitem__(self, index):
        # get the image and the label
        img, target = self.data[index], int(self.targets[index])

        # bounding box
        bbox = None

        # generate new image with randomly position and size
        if self.rand_mnist is not None:
            img, bbox = self.rand_mnist(img)

        # normalize the data to [0, 1]
        if self.norm_data is not None:
            img = self.norm_data(img, 0, 255, 1)

        # use YOLO grid system
        if self.grids_system is not None and bbox is not None:
            target = self.grids_system(target, bbox)

        return img, target

    def __len__(self):
        return super().__len__()


def test():
    # create the MNIST dataset
    dataset = MNISTDataset(root='../data/MNIST', train=True, download=False,
                           rand_mnist=GenerateRandMNIST(), grids_system=YoloGrids(), norm_data=generic_normalize)

    # create data loader
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    # iterate over the data
    for i, (img, target) in enumerate(data_loader):
        # print out the data dimensions
        print(img.shape, target.shape)

        # plot image
        plot_mnist_image(images=img, marks=target)

        # sum the batch
        total = img.sum()
        print(total)

        # break the loop
        if i == 10:
            break


if __name__ == '__main__':
    test()

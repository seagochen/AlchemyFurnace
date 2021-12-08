from typing import Callable, Optional

import cv2
import numpy as np
from torchvision import datasets

from CvTools import ImageConverter as converter
from GridSystem.YoloGrids import YoloGrids


class MNISTWrapperDataset(datasets.MNIST):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 yolo_grids: Optional[YoloGrids] = None,
                 img_size: tuple = (448, 448, 3),
                 obj_size: tuple = (64, 64)):

        # call super constructor
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # define the channels, height and width of the image
        self.img_size = img_size
        self.obj_size = obj_size

        # create the net grids
        self.grids = yolo_grids

    def __getitem__(self, index):
        # get the image and the label
        img, target = self.data[index], int(self.targets[index])

        # embed the object in the new image
        shift_x = np.random.randint(0, self.img_size[0] - self.obj_size[0])
        shift_y = np.random.randint(0, self.img_size[1] - self.obj_size[1])
        img = self.embed_obj(img, shift_x, shift_y)

        # and transform image if required
        if self.transform is not None:
            img = self.transform(img)

        # transform the label if required
        if self.target_transform is not None:
            target = self.target_transform(target)

        elif self.grids is not None:
            target = self.grids.set_yolo_target(target,
                                                shift_x, shift_y,
                                                self.obj_size[0] + shift_x,
                                                self.obj_size[1] + shift_y)

        return img, target

    def __len__(self):
        return super().__len__()

    def embed_obj(self, img: any, shift_x: int, shift_y: int) -> np.ndarray:

        # convert the PIL image to a numpy array
        img = converter.img_to_cv(img)

        # resize the image to 56x56
        img = cv2.resize(img, self.obj_size, interpolation=cv2.INTER_CUBIC)

        # create a new image with the given size
        blank_image = np.zeros(self.img_size, np.uint8)

        # we now need to put the picture in a random position of the new image
        blank_image[shift_y: shift_y + self.obj_size[0], shift_x: shift_x + self.obj_size[1]] = img

        # now we need to convert the blank image to PIL Image
        img = converter.cv_to_img(blank_image)

        return img


def test():
    data_dir = '../data/MNIST'
    dataset = MNISTWrapperDataset(root=data_dir, train=True, download=False)

    # show image
    for i in range(len(dataset)):
        img, target = dataset[i]
        img = converter.img_to_cv(img)

        print(target, img.shape)
        # cv2.imshow('img', converter.img_to_cv(img))
        # cv2.waitKey(0)


if __name__ == '__main__':
    test()

from typing import Callable, Optional

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import datasets

from CvTools import ImageConverter as converter
from CvTools.NetworkDetectedResult import NetworkDetectedResult


class MNISTWrapperDataset(datasets.MNIST):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 download: bool = False):
        super().__init__(root, train, transform, download)

        # define the channels, height and width of the image
        self.height = 224
        self.width = 224
        self.obj_size = (56, 56)

        # define the size of the grid
        self.grid_size_x = 8
        self.grid_size_y = 8
        self.bbox_size = 2
        self.categories = 10

        # use yolo grids to mark the object
        self.use_yolo_grids = True

        # create the net grids
        self.grids = NetworkDetectedResult(
            grids_cols=self.grid_size_x,
            grids_rows=self.grid_size_y,
            bounding_boxes=self.bbox_size,
            object_categories=self.categories)

    def __getitem__(self, index):
        # get the image and the label
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # convert the image and place it to a random position
        shift_x = np.random.randint(0, self.width - 56)
        shift_y = np.random.randint(0, self.height - 56)
        img = self.embed_obj(img, shift_x, shift_y)

        # and transform image if required
        if self.transform is not None:
            img = self.transform(img)

        # transform the target if required
        if self.use_yolo_grids:
            target = self.yolo_mark(target, shift_x, shift_y)

        return img, target

    def __len__(self):
        return super().__len__()

    def embed_obj(self, img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:

        # convert the PIL image to a numpy array
        img = np.asarray(img)

        # resize the image to 56x56
        img = cv2.resize(img, self.obj_size, interpolation=cv2.INTER_CUBIC)

        # create a new image with the given size
        blank_image = np.zeros((self.height, self.width), np.uint8)

        # we now need to put the picture in a random position of the new image
        blank_image[shift_y: shift_y + img.shape[0], shift_x: shift_x + img.shape[1]] = img

        # now we need to convert the blank image to PIL Image
        img = Image.fromarray(blank_image)

        return img

    def yolo_mark(self, target: int, shift_x: int, shift_y: int) -> torch.Tensor:
        # # clear the net first
        # self.net_grids.clear()
        #
        # # create grid dataset
        # result = grid_result(None,
        #                      bounding_boxes=self.bbox_size,
        #                      object_categories=self.categories)
        #
        # # set some dataset
        # result.set_confidence(0, 1)
        # result.set_object_category(target)
        #
        # # set bounding box
        # bbox = b_box(0, 0, self.obj_size[0], self.obj_size[1])
        # result.set_bounding_box(0, bbox.get_cornered_bbox())
        # result.set_bounding_box(1, bbox.get_cornered_bbox())
        #
        # # update the net
        # self.net_grids.update(result)

        return target


def test():
    data_dir = '../../data/MNIST'
    dataset = MNISTWrapperDataset(root=data_dir, train=True, download=False)

    # show image
    for i in range(len(dataset)):
        img, target = dataset[i]

        print(target)
        cv2.imshow('img', converter.img_to_cv(img))
        cv2.waitKey(0)


if __name__ == '__main__':
    test()

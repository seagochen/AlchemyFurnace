from typing import Callable, Optional

import cv2
import numpy as np
from torchvision import datasets

from YoloVer1.tools import ImageConvertor as converter
from YoloVer1.grids.YoloGrids import YoloGrids

from torch.utils.data import DataLoader
from torchvision import transforms


class MNISTDataset(datasets.MNIST):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 use_grids: Optional[Callable] = None,
                 img_size: tuple = (448, 448, 3),
                 obj_size: tuple = (64, 64)):

        # call super constructor
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # define the channels, height and width of the image
        self.img_size = img_size
        self.obj_size = obj_size

        # create the net grids
        self.grids_func = use_grids

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

        elif self.grids_func is not None:
            target = self.grids_func(target,
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


def test_without_grids():
    data_dir = "../data/MNIST"
    dataset = MNISTDataset(root=data_dir, train=True, download=True)

    # show image
    for i in range(10):
        img, target = dataset[i]
        img = converter.img_to_cv(img)

        cv2.imshow('img', converter.img_to_cv(img))
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def test_with_grids():
    batch_size = 10
    grids_size = (6, 6)
    confidences = 1
    bounding_boxes = 2
    object_categories = 10
    data_dir = "../data/MNIST"

    # define yolo grids
    grids = YoloGrids(grids_size=grids_size,
                      confidences=confidences,
                      bounding_boxes=bounding_boxes,
                      object_categories=object_categories)

    # transform sequential
    transform = transforms.Compose([
        transforms.ToTensor(),
        #                     mean       std
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load data
    dataset = MNISTDataset(root=data_dir, train=True, download=True, use_grids=grids, transform=transform)


    # training loader
    loader = DataLoader(dataset,
                              shuffle=True,
                              batch_size=batch_size)

    for batch_idx, (data, target) in enumerate(loader):
        print(data.shape, target.shape)

        if batch_idx > 100:
            break


if __name__ == '__main__':
    test_without_grids()
    test_with_grids()

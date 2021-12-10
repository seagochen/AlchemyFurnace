import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from SimpleMNISTDetection.grids.YoloGrids import YoloGrids
from CvTools import ImageConvertor as convertor

from SimpleMNISTDetection.dataset.ModifiedDataset import MNISTWrapperDataset


class YoloLoss(torch.nn.Module):

    def __init__(self, grids_size=(1, 1), confidences=0, boxes=0, categories=10,
                 lambda_coord=5, lambda_noobj=0.5, lambda_obj=1,
                 threshold_ignore=0.5, threshold_obj=0.5):
        super().__init__()

        # keep the parameters
        self.grids_size = grids_size
        self.confidences = confidences
        self.boxes = boxes
        self.categories = categories

        self.lamda_coord = lambda_coord
        self.lamda_noobj = lambda_noobj
        self.lamda_obj = lambda_obj
        self.threshold_ignore = threshold_ignore
        self.threshold_obj = threshold_obj

        # mean squared error loss
        self.mse_loss = torch.nn.MSELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        :param y_pred: shape of tensor is [B, C, W, H]
        :param y_true: shape of tensor is [B, C]
        """
        return None


def test():
    # global variables
    batch_size = 4
    grids_cols = 1
    grids_rows = 1
    confidences = 1
    bounding_boxes = 2
    object_categories = 10
    data_dir = '../../data/MNIST'

    # transform sequential
    transform = transforms.Compose([
        transforms.ToTensor(),
        #                     mean       std
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # use yolo grids
    yolo_grids = YoloGrids(grids_cols=grids_cols,
                           grids_rows=grids_rows,
                           confidences=confidences,
                           bounding_boxes=bounding_boxes,
                           object_categories=object_categories)

    # training dataset
    train_dataset = MNISTWrapperDataset(root=data_dir,
                                        train=True,
                                        download=True,
                                        # transform=transform,
                                        yolo_grids=yolo_grids)

    # training loader
    train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=batch_size)

    # loop the data
    for batch_idx, (data, target) in enumerate(train_loader):

        img = convertor.img_to_cv(data)

        print(target, img.shape)

        cv2.imshow('img', convertor.img_to_cv(img))
        cv2.waitKey(0)


if __name__ == "__main__":
    test()
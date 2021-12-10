import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from SimpleMNISTDetection.dataset.ModifiedDataset import MNISTWrapperDataset
from SimpleMNISTDetection.grids.YoloGrids import YoloGrids
from SimpleMNISTDetection.model.YoloNetwork import SimpleYoloNetwork


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
        :param y_pred: shape of tensor is [B, C]
        :param y_true: shape of tensor is [B, C]
        """

        # calculate the real dimensions of the grids
        B, C = y_pred.shape
        G = self.grids_size[0] * self.grids_size[1]
        C = C / G

        # reshape the truth and prediction tensors
        y_pred = y_pred.view(B, C, G)
        y_true = y_true.view(B, C, G)

        # calculate the loss
        loss = torch.zeros(B)

        # loop over the grids
        for i in range(G):
            pass

        return loss


def test():
    # global variables
    batch_size = 4
    grids_cols = 8
    grids_rows = 8
    confidences = 1
    bounding_boxes = 2
    object_categories = 10


if __name__ == "__main__":
    test()

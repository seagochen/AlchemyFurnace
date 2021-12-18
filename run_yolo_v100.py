import os

import torch
from torch.utils.data import DataLoader

from Generic.dataset.MNIST.MNISTDataset import MNISTDataset, GenerateRandMNIST
from Generic.grids.YoloGrids import YoloGrids
from Generic.tools.Normalizer import generic_normalize
from YoloVer100.model.YoloNetwork import YoloV1Network

# global variables
epochs = 30
batch_size = 1
grids_size = (8, 8)
confidences = 1
bounding_boxes = 1
object_categories = 10

# data folder
data_dir = 'data/MNIST'

# model folder
model_path = 'YoloVer100/model/yolo_v100.pth'

# training dataset
dataset = MNISTDataset(root=data_dir, train=True, download=True,
                       rand_mnist=GenerateRandMNIST(),
                       grids_system=YoloGrids(),
                       norm_data=generic_normalize)

# training loader
train_loader = DataLoader(dataset,
                          shuffle=True,
                          batch_size=batch_size)


def test(model, device, loader):

    # set model to test mode
    model.eval()

    # test the model
    with torch.no_grad():

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # forward only
            output = model(data)

            # display thee result



def run_display_demo():
    # define model
    model = YoloV1Network(grids_size=grids_size,
                          confidences=confidences,
                          bounding_boxes=bounding_boxes,
                          object_categories=object_categories)

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model parameters if exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("load model parameters successfully!")


if __name__ == "__main__":
    run_display_demo()
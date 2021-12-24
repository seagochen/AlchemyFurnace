import os

import torch
from torch.utils.data import DataLoader

from Generic.dataset.MNIST.MNISTDataset import MNISTDataset, GenerateRandMNIST
from Generic.grids.YoloGrids import YoloGrids
from Generic.tools.Normalizer import generic_normalize
from Generic.tools.ImagePlotter import mark_detected_obj
from Generic.grids.BoundingBox import yolo_coord
from Generic.tools.Convertor import *
from YoloVer100.model.YoloNetwork import YoloV1Network

# global variables
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


def cognition(data: torch.Tensor, label: torch.Tensor, threshold=0.5):
    # convert tensor to opencv image
    data_frame = tensor_to_cv(data)

    # derive the object confidences, bounding boxes, and object categories parts
    # from the label tensor
    confidences, bounding_boxes, object_categories = label[0, :], label[1:5, :], label[5:, :]

    # determine the maximum confidence and its index
    max_conf, max_index = torch.max(confidences, dim=0)

    # get the bounding box and object category
    bounding_box = bounding_boxes[:, max_index]
    object_category = object_categories[:, max_index]

    # get the maximum object value and index
    max_object_value, max_object_index = torch.max(object_category, dim=0)

    # get the object category name
    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    object_name = categories[max_object_index.item()]

    # if the maximum confidence is greater than the threshold
    if max_conf.item() > threshold:
        # get the coordinates of the bounding box
        cent_x, cent_y, width, height = bounding_box.tolist()
        grid_j = max_index.item() % 8
        grid_i = max_index.item() // 8

        # convert the yolo style coordinates to the real world coordinates
        lt_x, lt_y, rb_x, rb_y = yolo_coord((cent_x, cent_y, width, height, grid_i, grid_j), grids_size=grids_size,
                                            gamma=448)

        # round the floats to integers
        lt_x = int(round(lt_x))
        lt_y = int(round(lt_y))
        rb_x = int(round(rb_x))
        rb_y = int(round(rb_y))

        # mark the rect and the name on the image
        data_frame = mark_detected_obj(data_frame,
                                       text=object_name, text_coord=(lt_x, rb_y), font_size=1, font_color=(0, 255, 0),
                                       bbox_coord=(lt_x, lt_y, rb_x, rb_y), box_color=(0, 0, 255))

    else: # nothing to do
        print("failed to detect the object")

    return data_frame


def test(model, loader):
    # set model to test mode
    model.eval()

    # test the model
    with torch.no_grad():

        for data, target in loader:

            # forward only
            output = model(data)

            # display thee result
            for i in range(batch_size):

                # print the label with rect and recornized mark on the image
                # shape of the input data is [3, 448, 448]
                # shape of the label is [15, 64]
                image_frame = cognition(data[i], output[i])

                # display image
                cv2.imshow('image', image_frame)
                if cv2.waitKey(0) == ord('q'):
                    break


def run_display_demo():
    # define model
    model = YoloV1Network(grids_size=grids_size,
                          confidences=confidences,
                          bounding_boxes=bounding_boxes,
                          object_categories=object_categories)

    # load model parameters if exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("load model parameters successfully!")

    # test the training result
    test(model, train_loader)


if __name__ == "__main__":
    run_display_demo()

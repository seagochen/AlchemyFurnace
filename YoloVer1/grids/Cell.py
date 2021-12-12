import torch
from siki.numeric.Basics import *


class Cell(object):
    """
    This class is used to store the detected result of each grid.
    """

    def __init__(self, confidences=1, bounding_boxes=1, object_categories=1):
        super().__init__()

        # save the parameters
        self.data = None
        self.confidences = confidences
        self.bounding_boxes = bounding_boxes
        self.object_categories = object_categories

        self.conf_tensor = None
        self.bbox_tensor = None
        self.obj_tensor = None

    def focus_on(self, grid_data: torch.Tensor) -> None:
        """
        This function is used to focus the detection grid of the result.
        """

        # save the dataset
        self.data = grid_data.reshape(-1)

        # from the grid, we can obtained the confidence, bounding box and object category
        self.conf_tensor = self.data[:self.confidences]
        self.bbox_tensor = self.data[self.confidences:self.confidences + self.bounding_boxes * 4]
        self.obj_tensor = self.data[self.confidences + self.bounding_boxes * 4:]

    def get_object_category(self) -> tuple:
        # find the max value and index
        max_value, max_index = torch.max(self.obj_tensor, 0)
        return max_index, max_value

    def set_object_category(self, ind: int, obj_confidence: float = 1.) -> None:
        # zero the object category
        self.obj_tensor.zero_()

        # set the object category
        self.obj_tensor[ind] = obj_confidence

    def get_bounding_box(self, ind: int) -> torch.Tensor:
        # get the bounding box by index
        return self.bbox_tensor[ind * 4:ind * 4 + 4]

    def set_bounding_box(self, ind: int, bbox_pts: any) -> None:
        bbox = self.bbox_tensor[ind * 4:ind * 4 + 4]
        for i in range(4):
            bbox[i] = clamp(bbox_pts[i], 0., 1.)

    def get_confidence(self, ind=0) -> float:
        return self.conf_tensor[ind]

    def set_confidence(self, ind: int, value: float) -> None:
        self.conf_tensor[ind] = clamp(value, 0., 1.)

    def tensor(self):
        return self.data

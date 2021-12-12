import torch
from YoloVer1.grids.BoundingBox import grids_coord, yolo_coord


def set_value(tensor, grid_i, grid_j, cell_ind, value):
    tensor[cell_ind, grid_i, grid_j] = value


class YoloGrids(object):

    def __init__(self, grids_size: tuple = (8, 8), spatial_size: tuple = (448, 448), confidences: int = 1,
                 bounding_box: int = 1, object_categories: int = 10, alpha: float = 0., beta: float = 448.,
                 gamma: float = 1.):

        # keep parameters
        self.grids_size = grids_size
        self.spatial_size = spatial_size
        self.confidences = confidences
        self.bounding_box = bounding_box
        self.object_categories = object_categories
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, target: int, bounding_box: tuple) -> torch.Tensor:

        # generate coordinates
        coord = grids_coord(bounding_box, spatial_size=self.spatial_size, grids_size=self.grids_size,
                            alpha=self.alpha, beta=self.beta, gamma=self.gamma)

        # generate target in tensor format
        features = self.confidences + self.bounding_box * 4 + self.object_categories
        tensor = torch.zeros(features, self.grids_size[0], self.grids_size[1])

        # set target
        self.set_confidences(tensor, coord['grid_i'], coord['grid_j'])
        self.set_bounding_box(tensor, coord['grid_i'], coord['grid_j'],
                              (coord['cent_x_rel'], coord['cent_y_rel'],
                               coord["rb_x"] - coord["lt_x"], coord["rb_y"] - coord["lt_y"]))
        self.set_categories(tensor, coord['grid_i'], coord['grid_j'], target)

        # return target
        return tensor.reshape(-1, self.grids_size[0] * self.grids_size[1])

    def set_confidences(self, tensor: torch.Tensor, grid_i: int, grid_j: int):
        for i in range(self.confidences):
            set_value(tensor, grid_i, grid_j, i, 1)

    def set_bounding_box(self, tensor: torch.Tensor, grid_i: int, grid_j: int, bbox: tuple):
        for i in range(self.bounding_box):
            set_value(tensor, grid_i, grid_j, self.confidences + i * 4 + 0, bbox[0])
            set_value(tensor, grid_i, grid_j, self.confidences + i * 4 + 1, bbox[1])
            set_value(tensor, grid_i, grid_j, self.confidences + i * 4 + 2, bbox[2])
            set_value(tensor, grid_i, grid_j, self.confidences + i * 4 + 3, bbox[3])

    def set_categories(self, tensor: torch.Tensor, grid_i: int, grid_j: int, target: int):
        set_value(tensor, grid_i, grid_j, self.confidences + self.bounding_box * 4 + target, 1)


def test():
    grids = YoloGrids()
    target = grids(0, (2, 214, 66, 278))

    for g in range(64):
        print(target[:, g])


if __name__ == "__main__":
    test()

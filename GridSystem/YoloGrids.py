import torch
from CvTools.NetworkDetectedResult import NetworkDetectedResult
from CvTools import BoundingBox as bbox


class YoloGrids(NetworkDetectedResult):

    def __init__(self, grids_cols, grids_rows, confidences, bounding_boxes, object_categories,
                 spatial_size: tuple = (224, 224), alpha: float = 0., beta: float = 224., gamma: float = 1.):

        # super constructor
        super().__init__(grids_cols=grids_cols,
                         grids_rows=grids_rows,
                         confidences=confidences,
                         bounding_boxes=bounding_boxes,
                         object_categories=object_categories)

        # keep the parameters
        self.spatial_size = spatial_size
        self.grid_size = (grids_cols, grids_rows)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def set_yolo_target(self, target, left_x, top_y, right_x, bottom_y) -> torch.Tensor:
        # calculate the bounding box relative coordination
        general_coord = bbox.grids_coord((left_x, top_y, right_x, bottom_y),
                                         spatial_size=self.spatial_size,
                                         grid_size=self.grid_size,
                                         alpha=self.alpha, beta=self.beta, gamma=self.gamma)

        cent_x, cent_y = general_coord["cent_x_rel"], general_coord["cent_y_rel"]
        bbox_w, bbox_h = general_coord["rb_x"] - general_coord["lt_x"], general_coord["rb_y"] - general_coord["lt_y"]
        grid_i, grid_j = general_coord["grid_i"], general_coord["grid_j"]

        # update the grid's object category
        grid = self.focus_cursor(grid_i, grid_j)
        grid.set_object_category(target)

        # update the grid's confidence
        for i in range(self.confidences):
            grid.set_confidence(i, 1)

        # update the grid's bounding box coordination
        for i in range(self.bounding_boxes):
            grid.set_bounding_box(i, (cent_x, cent_y, bbox_w, bbox_h))

        # return the tensor of grids
        return self.tensor

    def set_yolo_pre(self, confidence, obj_ind, obj_confidence,
                     bbox_ind, cent_x, cent_y, bbox_w, bbox_h, grid_i, grid_j) -> torch.Tensor:
        grid = self.focus_cursor(grid_i, grid_j)

        # update the grid's object category
        grid.set_object_category(obj_ind, obj_confidence)

        # update the grid's confidence
        for i in range(self.confidences):
            grid.set_confidence(i, confidence)

        # update the grid's bounding box coordination
        grid.set_bounding_box(bbox_ind, (cent_x, cent_y, bbox_w, bbox_h))

        # return the tensor of grids
        return self.tensor

    def get_ltrb_coord(self, grid_i, grid_j, bbox_ind, gamma=224) -> tuple:
        # focus on grid
        grid = self.focus_cursor(grid_i, grid_j)

        # get the bbox coordinate
        yolo_coord = grid.get_bounding_box(bbox_ind)

        cent_x = yolo_coord[0]
        cent_y = yolo_coord[1]
        bbox_w = yolo_coord[2]
        bbox_h = yolo_coord[3]

        # convert the yolo coordinate to left-top-right-bottom coordinate
        ltrb_coord = bbox.yolo_coord((cent_x, cent_y, bbox_w, bbox_h, grid_i, grid_j),
                                     grid_size=self.grid_size, gamma=gamma)

        return ltrb_coord


def test():
    # Assume we have 8x8 grids, 10 object category, 2 bounding boxes for each grid and 1 confidence
    grids_cols = 8
    grids_rows = 8
    object_categories = 10
    bounding_boxes = 2
    confidences = 1

    # create the grids for yolo detection model
    target_grids = YoloGrids(grids_cols=grids_cols, grids_rows=grids_rows,
                             confidences=confidences,
                             object_categories=object_categories,
                             bounding_boxes=bounding_boxes)

    pre_grids = YoloGrids(grids_cols=grids_cols, grids_rows=grids_rows,
                          confidences=confidences,
                          object_categories=object_categories,
                          bounding_boxes=bounding_boxes)

    # set the target and pred
    target_grids.set_yolo_target(target=8, left_x=33, top_y=40, right_x=89, bottom_y=96)
    pre_grids.set_yolo_pre(confidence=0.9,
                           obj_ind=5, obj_confidence=0.3,
                           bbox_ind=0, cent_x=0.3, cent_y=0.3, bbox_w=0.5, bbox_h=0.5,
                           grid_i=2, grid_j=2)

    print(target_grids.get_ltrb_coord(2, 2, 0))
    print(target_grids.get_ltrb_coord(2, 2, 1))
    print(pre_grids.get_ltrb_coord(2, 2, 0))


if __name__ == "__main__":
    test()

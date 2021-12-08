import torch
from CvTools.GridDetectionResult import GridDetectionResult


class NetworkDetectedResult(object):
    """
    物体识别结果
    用统一的格式保存物体识别结果，避免由于算法不同或者处理疏忽导致的意外情况
    """

    def __init__(self, tensor_grids=None, grids_cols=1, grids_rows=1,
                 confidences=1, bounding_boxes=1, object_categories=1) -> None:
        """
        :param tensor_grids: 物体识别结果的tensor数据, 可以为空
        :param grids_cols: 物体识别网格的列数，不得为空，默认为1
        :param grids_rows: 物体识别网格的行数，不得为空，默认为1
        :param object_categories: 物体类别数，不得为空，默认为1
        :param bounding_boxes: 物体框数，不得为空，默认为1
        :param confidences: 置信度数，不得为空，默认为1
        """
        super().__init__()

        # basic info of the object detection tensor
        self.grids_cols = grids_cols
        self.grids_rows = grids_rows
        self.object_categories = object_categories
        self.confidences = confidences

        # counting the real size of bounding boxes
        self.bounding_boxes = bounding_boxes * 4  # 4 coordinates of bounding boxes

        # assign tensor dataset to the object detection result
        if tensor_grids is None:
            # shape of [confidences, bounding_boxes, objects, width (cols), height (rows)]
            self.grids = torch.zeros(
                self.confidences + self.bounding_boxes + self.object_categories,
                self.grids_cols, self.grids_rows, dtype=torch.float32)
        else:
            # convert the dataset type to float32
            self.grids = tensor_grids.float()

        # reshape the tensor dataset to [confidences x bounding_boxes x objects, width, height]
        self.grids = self.grids.reshape(-1, self.grids_cols, self.grids_rows)

        # set cursor to the grids
        self.cursor = GridDetectionResult(
            confidences=confidences,
            bounding_boxes=bounding_boxes,
            object_categories=object_categories)

        # check the tensor size
        self.check_tensor_size()

    def check_tensor_size(self) -> None:
        # check the dimensions of grids
        if self.grids_rows <= 0 or self.grids_cols <= 0:
            raise ValueError("grids_rows or grids_cols must not be zero!")

        # check the dimensions of object categories, bounding boxes and confidences
        if self.object_categories <= 0 or self.bounding_boxes <= 0 or self.confidences <= 0:
            raise ValueError("object_categories, bounding_boxes, confidences must not be zero!")

        # flatten the tensor to 1D and calculate the size
        flatten_size = self.grids.reshape(-1).shape[0]

        # check the flatten tensor size
        if flatten_size != self.grids_rows * self.grids_cols * \
                (self.confidences + self.bounding_boxes + self.object_categories):
            # raise value error
            raise ValueError("grid size is not correct!")

    def focus(self, grid_x: int = 0, grid_y: int = 0) -> GridDetectionResult:
        # get the grid from the tensor dataset
        self.cursor.focus_on(self.grids[:, grid_x, grid_y])
        return self.cursor

    def clear(self) -> None:
        """
        清空网格数据
        """
        # clear the tensor dataset
        self.grids.fill_(0)


def test():
    # Assume we have 8x8 grids, 10 object category, 2 bounding boxes for each grid and 1 confidence
    grids_cols = 8
    grids_rows = 8
    object_categories = 10
    bounding_boxes = 1
    confidences = 1

    # create a grids tensor
    grids = NetworkDetectedResult(grids_cols=grids_cols, grids_rows=grids_rows,
                                  confidences=confidences, bounding_boxes=bounding_boxes,
                                  object_categories=object_categories)
    # get grid from grids
    grid = grids.focus(2, 2)

    # update the dataset
    grid.set_confidence(0, 11)
    grid.set_object_category(5)
    grid.set_bounding_box(0, (0, -9, 1.5, 1.5))

    # print out the tensor dataset
    for x in range(grids_cols):
        for y in range(grids_rows):
            # get grid from grids
            grid = grids.focus(x, y)

            # print grids
            print("grid: {}, {}, confidence: {}, object_category: {}, bounding_box: {}".format(
                x, y, grid.get_confidence(0), grid.get_object_category(), grid.get_bounding_box(0)))

        print("")


if __name__ == "__main__":
    test()

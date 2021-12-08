import torch
from yolo.Dataset.YoLoImageDescription import YoLoImageInfo


def _fill_ground_truth(tensor: torch.Tensor, cx, cy, width, height, class_type):
    # 填充第一个bbox
    tensor[0] = cx
    tensor[1] = cy
    tensor[2] = width
    tensor[3] = height
    tensor[4] = 1.  # 这里表示该网格有数据，还不是IOU

    # 填充第二个bbox
    tensor[5] = cx
    tensor[6] = cy
    tensor[7] = width
    tensor[8] = height
    tensor[9] = 1.  # 把上述数据重复一遍，为的是能够方便之后的LOSS计算

    # 识别类型的编码
    tensor[10 + class_type] = 1.

    return tensor


def compute_ground_truth(info: YoLoImageInfo, bbox=2, grid_cells=7, classes=20):
    """
    模型预测得出的张量数据结构如下：
    [7 x 7][x1, y1, w1, h1, Pr1(Object), x2, y2, w2, h2, Pr2(Object), Pr(Class_i | Object)]

    因此，为了计算损失函数，我们需要根据YoloImageInfo生成类似的数据结构

    :param info:
    :param bbox:
    :param grid_cells:
    :param classes:
    :return:
    """

    # 创建一个 [7, 7, 5 x bbox + classes] 的张量
    ground_truth = torch.zeros(grid_cells, grid_cells, 5 * bbox + classes).float()

    # 开始遍历 info 中存在的每个 Object
    for obj in info.objects:
        # 物体所在的网格坐标
        i, j = obj.grid_i, obj.grid_j

        # bounding box 中心、长宽等信息
        cx, cy, w, h = obj.cent_x, obj.cent_y, obj.bbox_w, obj.bbox_h

        # 物体类型
        class_type = obj.class_type

        # 把数据写入网格中
        ground_truth[i, j, :] = _fill_ground_truth(
            tensor=ground_truth[i, j, :],
            cx=cx, cy=cy, width=w, height=h, class_type=class_type)

    # 返回
    return ground_truth

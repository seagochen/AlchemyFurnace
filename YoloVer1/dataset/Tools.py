import numpy as np
import torch
from YoloVer1.grids.BoundingBox import yolo_coord


def round_pts(pts):
    """
    Round the points.
    :param pts:
    :return:
    """
    return [int(np.round_(pt, 1)) for pt in pts]


def derive_bounding_box(yolo_grids: torch.Tensor,
                        threshold: float = 0.5,
                        grids_size: tuple = (8, 8),
                        confidences: int = 1,
                        bounding_boxes: int = 1, 
                        gamma: int = 448):
    """
    According to the yolo_grids' confidence, derive the bounding box.
    Generally, the chosen box is 
    :param gamma:
    :param confidences:
    :param grids_size:
    :param bounding_boxes:
    :param threshold:
    :param yolo_grids:
    :return:
    """

    last_conf = 0
    last_box = None

    # 遍历每个网格
    for i in range(grids_size[0]):
        for j in range(grids_size[1]):
            
            # 计算网格位移量
            ind = i + j * grids_size[0]

            # 获取当前网格的置信度
            conf = yolo_grids[:confidences, ind].item()

            # 如果当前网格的置信度大于阈值，则计算当前网格的坐标
            if conf > threshold and conf > last_conf:

                # 更新置信度
                last_conf = conf

                # 计算bbox
                coord = yolo_grids[confidences: confidences + bounding_boxes * 4, ind].numpy()

                # 把坐标信息从中心点转换为左上、右下角点坐标
                last_box = yolo_coord((
                    coord[0], coord[1], coord[2], coord[3], i, j
                ), grid_size=grids_size, gamma=gamma)

    last_box = round_pts(last_box)
    return last_box


def derive_object_name(yolo_grids: torch.Tensor, labels: list,
                       threshold: float = 0.5,
                       grids_size: tuple = (8, 8),
                       confidences: int = 1,
                       bounding_boxes: int = 1,
                       object_categories: int = 10):
    """
    According to the object confidence of yolo grids, derive the object name.
    :param object_categories:
    :param bounding_boxes:
    :param confidences:
    :param grids_size:
    :param yolo_grids:
    :param labels:
    :param threshold:
    :return:
    """
    last_conf = 0
    last_text = None

    # 遍历每个网格
    for i in range(grids_size[0]):
        for j in range(grids_size[1]):
            
            # 计算网格位移量
            ind = i * grids_size[1] + j

            # 获取当前网格的置信度
            conf = yolo_grids[:confidences, ind].item()

            # 如果当前网格的置信度大于阈值，则计算当前网格的坐标
            if conf > threshold and conf > last_conf:

                # 更新置信度
                last_conf = conf

                # 取出当前网格的类别
                objects = yolo_grids[confidences + bounding_boxes * 4:, ind]

                # 找出最大的类别
                max_conf, max_ind = torch.max(objects, 0)

                # 获取类别名称
                last_text = labels[max_ind.item()]

                # 确定置信度
                if max_conf.item() < threshold:
                    last_text = "Unknown"

    return last_text

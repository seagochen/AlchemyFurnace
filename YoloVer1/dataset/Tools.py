import torch


def derive_bounding_box(yolo_grids: torch.Tensor, threshold: float = 0.5):
    """
    According to the yolo_grids' confidence, derive the bounding box.
    Generally, the chosen box is 
    :param threshold:
    :param yolo_grids:
    :return:
    """
    return None


def derive_object_name(yolo_grids: torch.Tensor, labels: list, threshold: float = 0.5):
    """

    :param yolo_grids:
    :param labels:
    :param threshold:
    :return:
    """
    return None
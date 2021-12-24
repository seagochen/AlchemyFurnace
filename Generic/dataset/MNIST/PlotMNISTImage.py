from typing import Callable, Optional

import cv2
import numpy as np
import torch

from Generic.grids.BoundingBox import yolo_coord
from Generic.tools.ImagePlotter import mark_detected_obj


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
            ind = i * grids_size[1] + j

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
                ), grids_size=grids_size, gamma=gamma)

    last_box = round_pts(last_box)
    return last_box


def derive_object_name(yolo_grids: torch.Tensor, labels: list,
                       threshold: float = 0.5,
                       grids_size: tuple = (8, 8),
                       confidences: int = 1,
                       bounding_boxes: int = 1):
    """
    According to the object confidence of yolo grids, derive the object name.
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
                if max_conf < threshold:
                    last_text = "Unknown"
                else:
                    last_text = labels[max_ind.item()]

    return last_text


def plot_mnist_image(images: torch.Tensor, marks: torch.Tensor,
                     img_transform: Optional[Callable] = None,
                     res_transform: Optional[Callable] = None) -> None:
    """
    把计算结果绘制到图片上
    :param images: 如果图片已经被归一化，要想正确显示回原来的信息，需要同时赋予 img_transform 函数
    :param marks: 必须使用 YOLO Grids 格式的计算结果，不能使用原始的MNIST label；如果数据已经被归一化了，需要同时指定 res_transform
    :param img_transform: 对图片的还原函数
    :param res_transform: 对结果的还原函数
    :return:
    """

    # transform the image back to range of [0, 255]
    if img_transform is not None:
        images = img_transform(images)

    # transform the result if necessary
    if res_transform is not None:
        marks = res_transform(marks)

    # detect the image dimensions
    if len(images.shape) != 4:
        raise ValueError("The input images must be 4-D tensor in shape of [B, C, W, H].")

    # get the shape of images
    B, C, H, W = images.shape

    # if the image is grayscale, convert it to RGB
    if C < 3:
        images = torch.cat([images, images, images], dim=1)

    # iterate over all the images
    for i in range(B):
        # first, we need to convert the tensor to opencv image
        img = images[i, :, :, :].numpy().transpose(1, 2, 0)

        # convert RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        """
        注意，我们假设一张图片就一个目标，所以这里显示的单目标情况，如果在新版本的网格中，需要一张图片多个目标时，需要修改这里的代码
        """

        # then, derive the bounding box coordinates
        bbox = derive_bounding_box(marks[i, :, :])

        # then, derive the label text from the mark
        label = derive_object_name(marks[i, :, :], labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        # draw the final image
        if bbox is not None and label is not None:
            marked_img = mark_detected_obj(image=img,
                                           text=label, text_coord=(bbox[0], bbox[3]), font_size=1,
                                           font_color=(0, 255, 255),
                                           bbox_coord=bbox, box_color=(255, 0, 0))
        else:
            marked_img = img

        # show image
        cv2.imshow("result", marked_img)

        # wait for key press
        cv2.waitKey(0)

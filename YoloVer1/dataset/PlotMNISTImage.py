from typing import Callable, Optional

import torch
import cv2

from YoloVer1.dataset import Tools as tools
from YoloVer1.tools.ImagePlotter import mark_detected_obj


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
        bbox = tools.derive_bounding_box(marks[i, :, :])

        # then, derive the label text from the mark
        label = tools.derive_object_name(marks[i, :, :], labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        print("transform data is", label, "at", bbox)

        # draw the final image
        if bbox is not None and label is not None:
            marked_img = mark_detected_obj(image=img,
                                           text=label, text_pt=(bbox[0], bbox[3]), font_size=2,
                                           font_color=(0, 255, 255),
                                           bbox=bbox, box_color=(255, 0, 0))
        else:
            marked_img = img

        # show image
        cv2.imshow("result", marked_img)

        # wait for key press
        cv2.waitKey(0)

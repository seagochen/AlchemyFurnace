import cv2
import numpy as np


def show_object_rect(image: any, bbox: tuple, color: tuple):
    """
    Display rectangle on image
    :param color:
    :param image:
    :param bbox: (left_x, left_y, right_x, right_y)
    :return:
    """

    pt1 = bbox[:2]
    pt2 = bbox[2:]
    image_out = image

    return cv2.rectangle(
        image_out,  # image
        pt1,  # left top pt
        pt2,  # right down pt
        color)  # color


def show_object_name(image: any, text: str, bbox: tuple, color: tuple):
    """
    Display text on image
    :param color:
    :param image:
    :param text:
    :param bbox: (left_x, left_y, right_x, right_y)
    :return:
    """
    pt1 = bbox[:2]
    pt2 = bbox[2:]

    # calculate left-top: x left_down: y of bonding box
    cx = pt1[0]
    cy = pt2[1]

    image_copy = image.copy()

    cv2.putText(
        image_copy,  # image
        text,  # text
        (cx, cy),  # orientation
        1,  # font
        2,  # font scale
        color)  # color

    return image_copy
    # return image.astype(np.uint8)


def show_object_name_with_rect(image: any, text: str, bbox: tuple, color: tuple):
    """
    Display text on image
    :param color:
    :param image:
    :param text:
    :param bbox: (left_x, left_y, right_x, right_y)
    :return:
    """

    # show name with rect
    image = show_object_name(image=image, text=text, bbox=bbox, color=color)
    image = show_object_rect(image=image, bbox=bbox, color=color)
    return image

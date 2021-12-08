import PIL.Image
import cv2

from yolo.Utils.ImgConvertor import img_to_cv, pad_img_align_left_top


def _update_annot_with_new_size(annot: dict, width, height, scalar_x: float, scalar_y: float):
    annot['size']['width'] = str(width)
    annot['size']['height'] = str(height)

    # modify coordinate information of each object in image
    for obj in annot['object']:
        xmin = float(obj['bndbox']['xmin']) * scalar_x
        ymin = float(obj['bndbox']['ymin']) * scalar_y

        xmax = float(obj['bndbox']['xmax']) * scalar_x
        ymax = float(obj['bndbox']['ymax']) * scalar_y

        # update object's coordinate information with new value
        obj['bndbox']['xmin'] = str(int(round(xmin)))
        obj['bndbox']['ymin'] = str(int(round(ymin)))
        obj['bndbox']['xmax'] = str(int(round(xmax)))
        obj['bndbox']['ymax'] = str(int(round(ymax)))

    return annot


def undistorted_resize_img(
        image: PIL.Image, annot: dict,
        target_width=448, target_height=448,
        max_padding_width=500, max_padding_height=500):

    # convert img to cv image
    image = img_to_cv(image)

    # pad the image
    image = pad_img_align_left_top(image, max_padding_width, max_padding_height)

    # calculate the shrink scalar
    scalar_x = target_width / max_padding_width
    scalar_y = target_height / max_padding_height

    # resize the image
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # modify the annot
    annot = _update_annot_with_new_size(annot, target_width, target_height, scalar_x, scalar_y)

    return image, annot


def distorted_resize_img(
        image: PIL.Image, annot: dict,
        target_width=448, target_height=448):

    # convert img to cv image
    image = img_to_cv(image)

    height, width, _ = image.shape

    # calculate the shrink scalar
    scalar_x = target_width / width
    scalar_y = target_height / height

    # resize the image
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # modify the annot
    annot = _update_annot_with_new_size(annot, target_width, target_height, scalar_x, scalar_y)

    return image, annot

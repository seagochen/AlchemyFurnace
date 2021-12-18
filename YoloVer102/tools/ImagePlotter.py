import cv2
import numpy as np


def mark_detected_obj(image: np.ndarray,
                      text: str, text_pt: [tuple, list], font_size: int, font_color: [tuple, list],
                      bbox: [tuple, list], box_color: [tuple, list]):
    """
    绘制带 bounding box 的图片
    :param image: BGR 彩色图片
    :param text: 绘制的文字
    :param text_pt: 文字的左上角坐标
    :param font_size: 字体大小
    :param font_color: 字体颜色
    :param bbox: bounding box 坐标（左上角坐标，右下角坐标）
    :param box_color: bounding box 颜色
    :return:
    """

    # copy image to avoid modify original image
    image_copy = image.copy()

    # calculate left-top: x left_down: y of bonding box
    pt1 = bbox[:2]
    pt2 = bbox[2:]

    # draw bounding box with default line width 2
    image_copy = cv2.rectangle(image_copy, pt1, pt2, box_color, 2)

    # draw text on image
    image_copy = cv2.putText(image_copy, text, text_pt, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 2)

    # return image
    return image_copy


def test():
    image = cv2.imread("../data/images/test.jpg")

    if image is None:
        print("Image not found")

    # mark detected object
    res = mark_detected_obj(image, "test", (10, 10), 1, (0, 0, 255), (10, 10, 100, 100), (0, 255, 0))

    # show image
    cv2.imshow("image", res)

    # wait for key press
    cv2.waitKey(0)


if __name__ == "__main__":
    test()

import cv2
import numpy as np


def mark_detected_obj(image: np.ndarray,
    text: str, text_pt: tuple, font_size: int, font_color: tuple,
    bbox: tuple, box_color: tuple):
    """
    Mark detected object on image
    :param color: (B, G, R)
    :param image: cv image with shape (H, W, C) and dtype=np.uint8
    :param text: text to display
    :param text_pt: font orientation (x, y) to display
    :param font_size: font size
    :param font_color: font color with (B, G, R) format
    :param bbox: bounding box with (left_x, left_y, right_x, right_y)
    :param box_color: bounding box color with color channel in (B, G, R) format
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
    image = cv2.imread("../dataset/images/test.jpg")

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
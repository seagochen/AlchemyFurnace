import cv2
import PIL
import numpy as np

import torch
from torchvision import transforms

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


def img_to_cv(image: PIL.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def cv_to_img(image: np.ndarray) -> PIL.Image:
    return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def tensor_to_cv(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img


def cv_to_tensor(image: np.ndarray) -> torch.Tensor:
    assert type(image) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(image))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256


def img_to_tensor(image: PIL.Image) -> torch.Tensor:
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)


def tensor_to_img(tensor: torch.Tensor) -> PIL.Image:
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def is_pil_img(image: any):
    return isinstance(image, PIL.Image)


def is_cv_img(image: any):
    return isinstance(image, np.ndarray)


def pad_img_align_left_top(image: np.ndarray, new_width: int, new_height: int, default_color=(0, 0, 0)):
    # get size of original image
    height, width, channels = image.shape

    # create an empty image
    new_img = np.full((new_height, new_width, channels), default_color, dtype=np.uint8)

    # copy original image and align to the left top corner to new image
    new_img[: height, : width] = image

    # return
    return new_img


def pad_img_align_center(image: np.ndarray, new_width: int, new_height: int, default_color=(0, 0, 0)):
    # get size of original image
    height, width, channels = image.shape

    # create an empty image
    new_img = np.full((new_height, new_width, channels), default_color, dtype=np.uint8)

    # compute center offset
    x_center = (new_width - width) // 2
    y_center = (new_height - height) // 2

    # copy image into center of result image
    new_img[y_center: y_center + height, x_center: x_center + width] = image

    # return
    return new_img

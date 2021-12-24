import cv2
import torch
import numpy as np

from PIL import Image

from torchvision import transforms

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

# 使用transforms函数把tensor转换成PIL.Image
unloader = transforms.ToPILImage()


def img_to_cv(image: Image) -> np.ndarray:
    """
    把PIL.Image转换成cv2.ndarray

    :param image: image要求是RGB格式
    :return: 转化后的图片是BGR格式
    """
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def cv_to_img(image: np.ndarray) -> Image:
    """
    把cv2.ndarray转换成PIL.Image

    :param image: image要求是BGR格式
    :return: 转化后的图片是RGB格式
    """
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def tensor_to_cv(tensor: torch.Tensor) -> np.ndarray:
    """
    把torch.Tensor转换成cv2.ndarray

    :param tensor: tensor的维度必须是[3, H, W]，颜色通道顺序无要求，数值范围是[0, 1]
    :return: 转化后是BGR格式的CV图片
    """
    # 先乘以255，然后再转化为uint8
    img = tensor.mul(255).byte()  

    # 把维度信息从[3, H, W]转换成[H, W, 3]
    img = img.numpy().transpose((1, 2, 0)) 

    # 再转化为BGR格式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 返回
    return img


def cv_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    把cv2.ndarray转换成torch.Tensor

    :param image: image的维度必须是[H, W, 3]，颜色通道顺序是BGR，数值范围是[0, 255]
    :return: 转化后是Tensor数据，维度是[3, H, W]，数值范围是[0, 1]
    """
    # 先对数据类型进行判断
    assert type(image) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(image))

    # 把颜色通道顺序转换成RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 把维度信息从[H, W, 3]转换成[3, H, W]
    img = torch.from_numpy(img.transpose((2, 0, 1)))

    # 再转化为Tensor数据，数值范围调整到[0, 1]
    return img.float().div(255)


def is_pil_img(image: any) -> bool:
    """
    判断是否是PIL.Image
    """
    return isinstance(image, Image)


def is_cv_img(image: any) -> bool:
    """
    判断是否是cv2.ndarray
    """
    return isinstance(image, np.ndarray)


def is_tensor(data: any) -> bool:
    """
    判断是否是torch.Tensor
    """
    return isinstance(data, torch.Tensor)


if __name__ == '__main__':
    data_a = np.random.randint(0, 255, size=(20, 30, 3), dtype=np.uint8)
    tensor_a = cv_to_tensor(data_a)

    img = tensor_to_cv(tensor_a)
    tensor_b = cv_to_tensor(img)

    c, h, w = tensor_a.size()
    for ci in range(c):
        for hi in range(h):
            for wi in range(w):
                print('a[{}, {}, {}] = {}'.format(ci, hi, wi, tensor_a[ci, hi, wi]), end='\t')
                print("b[{}, {}, {}] = {}".format(ci, hi, wi, tensor_b[ci, hi, wi]), end="\n")
            print("\n")
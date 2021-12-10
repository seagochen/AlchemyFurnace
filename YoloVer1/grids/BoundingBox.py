import math
from typing import Dict, Union, Any


def val_mapping(pts: tuple, alpha=0., beta=1., gamma=1.) -> tuple:
    """
    $$
    \begin{cases}
    \frac{pts}{(\beta - \alpha)} * \gamma
    \end{cases}
    $$
    """
    return tuple(
        pt / (beta - alpha) * gamma for pt in pts
    )


def val_clamp(pts: tuple, alpha=0., beta=1.) -> tuple:
    clamp_pts = lambda pt: pt if alpha <= pt <= beta else alpha if pt < alpha else beta
    return tuple(
        clamp_pts(pt) for pt in pts
    )


def grids_coord(rect_pts: tuple, spatial_size: tuple, grids_size: tuple, alpha=0., beta=1., gamma=1.) -> Dict[
    str, Union[int, Any]]:
    """
    将 bounding box 转化为网格坐标。

    :param rect_pts: 使用bounding box的左上角和右下角坐标，类型为tuple(ltx, lty, rbx, rby)
    :param spatial_size: 图像的空间尺寸，类型为tuple(width, height)
    :param grids_size: 网格尺寸，类型为tuple(x grids, y grids)
    :param alpha: 映射的起始值
    :param beta: 映射的终止值
    :param gamma: 映射的缩放系数
    :return: 网格坐标信息，类型为map
    """

    # 先确定bbox的中心点坐标
    center_x, center_y = (rect_pts[0] + rect_pts[2]) / 2, (rect_pts[1] + rect_pts[3]) / 2

    # 再确定grids的大小
    grid_width, grid_height = spatial_size[0] / grids_size[0], spatial_size[1] / grids_size[1]

    # 确定中心点的坐标
    grid_x, grid_y = math.ceil(center_x / grid_width), math.ceil(center_y / grid_height)
    grid_x = int(grid_x - 1)
    grid_y = int(grid_y - 1)

    # 确定中心点所在网格的相对坐标
    grid_x_rel, grid_y_rel = center_x - grid_x * grid_width, center_y - grid_y * grid_height
    grid_x_rel = grid_x_rel / grid_width
    grid_y_rel = grid_y_rel / grid_height

    # 数值坐标映射
    temp = val_mapping((
        rect_pts[0], rect_pts[1], rect_pts[2], rect_pts[3],
        center_x, center_y,
    ), alpha, beta, gamma)

    # 返回
    return {"lt_x": temp[0], "lt_y": temp[1], "rb_x": temp[2], "rb_y": temp[3],
            "cent_x_abs": temp[4], "cent_y_abs": temp[5],
            "cent_x_rel": grid_x_rel, "cent_y_rel": grid_y_rel,
            "grid_i": grid_x, "grid_j": grid_y}


def yolo_coord(pts: any, grid_size: tuple, alpha=0., beta=1., gamma=1.) -> tuple:
    """
    对于类似于YOLO的bbox表示方式（网格引索，相对中心点坐标，bbox长宽）转化为一般的bbox坐标（左上角坐标，右下角坐标）

    :param pts: YOLO bbox (cent_x, cent_y, width, height, grid_i, grid_j)
    :param grid_size: 网格尺寸，类型为tuple(x grids, y grids)
    :param alpha: 映射的起始值
    :param beta: 映射的终止值
    :param gamma: 映射的缩放系数
    :return: 通用bbox坐标信息 (lt_x, lt_y, rb_x, rb_y)
    """

    # 计算中心点的绝对坐标
    cent_x, cent_y = pts[0] + pts[4], pts[1] + pts[5]
    cent_x /= grid_size[0]
    cent_y /= grid_size[1]

    # 左上角的绝对坐标
    lt_x, lt_y = cent_x - pts[2] / 2, cent_y - pts[3] / 2

    # 右下角的绝对坐标
    rb_x, rb_y = cent_x + pts[2] / 2, cent_y + pts[3] / 2

    # 数值坐标映射
    temp = val_mapping((lt_x, lt_y, rb_x, rb_y), alpha, beta, gamma)

    # 返回
    return temp


if __name__ == "__main__":
    res = grids_coord((0, 0, 50, 50), (100, 100), (10, 10), alpha=0, beta=100, gamma=1)
    print(res)

    temp = yolo_coord((
        res["cent_x_rel"], res["cent_y_rel"],
        res["rb_x"] - res["lt_x"], res["rb_y"] - res["lt_y"],
        res["grid_i"], res["grid_j"]
    ), grid_size=(10, 10), gamma=100)
    print(temp)

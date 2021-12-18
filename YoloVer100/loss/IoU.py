import torch

GRIDS_NUM = 8


def intersection_of_length(pred_axis, pred_len, true_axis, true_len):
    """
    判断方法采用圆相交的判定规则，并且只计算单边相交长度

    :param pred_axis:
    :param pred_len:
    :param true_axis:
    :param true_len:
    :return:
    """
    # 计算半径
    pred_radius = pred_len / 2 * GRIDS_NUM
    true_radius = true_len / 2 * GRIDS_NUM

    # 计算半径的坐标点
    l_pred = pred_axis - pred_radius
    r_pred = pred_axis + pred_radius
    l_true = true_axis - true_radius
    r_true = true_axis + true_radius

    # 防止范围超出[0, 1]
    l_pred = torch.clamp(l_pred, min=0, max=GRIDS_NUM)
    r_pred = torch.clamp(r_pred, min=0, max=GRIDS_NUM)
    l_true = torch.clamp(l_true, min=0, max=GRIDS_NUM)
    r_true = torch.clamp(r_true, min=0, max=GRIDS_NUM)

    # 计算相交长度
    inter_len = torch.min(r_pred, r_true) - torch.max(l_pred, l_true)

    # 返回相交长度
    return inter_len / GRIDS_NUM


def compute_intersection_area(pred, true):
    # 得到pred与true关于x轴的相交长度
    inter_len_x = intersection_of_length(pred[:, 0, :], pred[:, 2, :], true[:, 0, :], true[:, 2, :])

    # 得到pred与true关于y轴的相交长度
    inter_len_y = intersection_of_length(pred[:, 1, :], pred[:, 3, :], true[:, 1, :], true[:, 3, :])

    # 计算相交面积
    inter_area = inter_len_x * inter_len_y

    # 返回相交面积
    return inter_area


def compute_union_area(pred, true, inter_area):
    # compute the box area of the pred
    area_pred = pred[:, 2, :] * pred[:, 3, :]
    area_true = true[:, 2, :] * true[:, 3, :]

    # compute the box area of the union
    area_union = area_pred + area_true - inter_area

    # return the area of the union
    return area_union


def compute_iou(pred, true):
    # get the size
    B, _, N = pred.size()

    # clamp the prediction and ground truth to be in [0, 1]
    pred = torch.clamp(pred, min=0, max=1)
    true = torch.clamp(true, min=0, max=1)

    # compute the intersection
    intersection = compute_intersection_area(pred, true)

    # compute the union
    union = compute_union_area(pred, true, intersection)

    # compute the IoU
    iou = intersection / (union + 1e-7)

    # finally, return the IoU
    return iou.reshape(B, N)


def test():
    # create a batch of boxes
    boxes_true = torch.tensor([
        [0.5, 0.5, 0.1, 0.1],
        [0.3, 0.3, 0.4, 0.4],
        [0.3, 0.3, 0.4, 0.4]
    ]).transpose(0, 1).reshape(1, 4, -1)

    boxes_pred = torch.tensor([
        [0.5, 0.5, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.0],
        [0.9, 0.9, 0.4, 0.4]
    ]).transpose(0, 1).reshape(1, 4, -1)

    # compute the IoU
    iou = compute_iou(boxes_pred, boxes_true)
    print(iou)


if __name__ == "__main__":
    test()

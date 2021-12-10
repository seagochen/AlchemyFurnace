import torch


def compute_union_area(box_pred, box_true):
    """
    Computes the union area between two boxes.
    :param box_pred: Tensor of shape (N, 4) containing the predicted boxes.
    :param box_true: Tensor of shape (N, 4) containing the true boxes.
    :return: Tensor of shape (N) containing the union area for each box.
    """

    # compute the left and right x coordinates of the union
    left = torch.min(box_pred[:, 0], box_true[:, 0])
    right = torch.max(box_pred[:, 2], box_true[:, 2])

    # compute the top and bottom y coordinates of the union
    top = torch.min(box_pred[:, 1], box_true[:, 1])
    bottom = torch.max(box_pred[:, 3], box_true[:, 3])

    # return the union area
    return (right - left) * (bottom - top)


def compute_intersection_area(box_pred, box_true):
    """
    Computes the intersection area between two boxes.
    :param box_pred: Tensor of shape (N, 4) containing the predicted boxes.
    :param box_true: Tensor of shape (N, 4) containing the true boxes.
    :return: Tensor of shape (N) containing the intersection area for each box.
    """

    # compute the left and right x coordinates of the intersection
    left = torch.max(box_pred[:, 0], box_true[:, 0])
    right = torch.min(box_pred[:, 2], box_true[:, 2])

    # compute the top and bottom y coordinates of the intersection
    top = torch.max(box_pred[:, 1], box_true[:, 1])
    bottom = torch.min(box_pred[:, 3], box_true[:, 3])

    # return the intersection area
    return (right - left) * (bottom - top)


def intersection_check(boxes_pre, boxes_true):
    """
    Check the boxes are really intersected.
    :param boxes_pre: Tensor of shape (N, 4) containing the predicted boxes.
    :param boxes_true: Tensor of shape (N, 4) containing the true boxes.
    :return: Tensor of shape (N) containing the intersection area for each box.
    """

    # concatenate the x coordinates of the boxes
    x_pts = torch.stack((boxes_pre[:, 0], boxes_pre[:, 2], boxes_true[:, 0], boxes_true[:, 2]), dim=1)

    # sort the x coordinates of the boxes
    x_pts = torch.sort(x_pts, dim=1)[0]

    # 框重合的情况
    mask = (x_pts[:, 0] == x_pts[:, 1]) & (x_pts[:, 2] == x_pts[:, 3])

    # 框相交的情况
    mask = mask | (x_pts[:, 0] == boxes_pre[:, 0]) & (x_pts[:, 1] == boxes_true[:, 0])
    mask = mask | (x_pts[:, 0] == boxes_true[:, 0]) & (x_pts[:, 1] == boxes_pre[:, 0])

    # return the mask
    return mask


def filter_iou(mask, iou):
    """
    Filters the IoU tensor by the mask.
    :param mask: Tensor of shape (B, N) containing the mask.
    :param iou: Tensor of shape (B, N) containing the IoU.
    :return: Tensor of shape (B, N) containing the filtered IoU.
    """
    # and then we need to reshape the mask and iou tensors
    mask = mask.reshape(-1)
    iou = iou.reshape(-1)

    # return the filtered IoU
    return torch.where(mask, iou, torch.zeros_like(iou))


def compute_iou_batch(boxes_pred, boxes_true):
    """
    Computes the Intersection over Union (IoU) for a batch of boxes.
    :param boxes_pred: Tensor of shape (B, N, 4) containing the predicted boxes.
    :param boxes_true: Tensor of shape (B, N, 4) containing the true boxes.
    :return: Tensor of shape (B, N) containing the IoU for each image in the batch.
    """

    # keep the dimensions of the input tensors
    B, N, _ = boxes_pred.shape  # B = batch size, N = number of boxes

    # to prevent the dataset leak, we need to use the detach function
    # and reshape the boxes_pred and boxes_true tensors
    boxes_pred = boxes_pred.detach().reshape(B * N, 4)  # left x, top y, right x, bottom x
    boxes_true = boxes_true.detach().reshape(B * N, 4)  # left x, top y, right x, bottom x

    # now we should also prevent the dataset out of the range [0, 1]
    boxes_pred = torch.clamp(boxes_pred, 0., 1.)
    boxes_true = torch.clamp(boxes_true, 0., 1.)

    # compute the intersection area
    intersection = compute_intersection_area(boxes_pred, boxes_true)

    # compute the union area
    union = compute_union_area(boxes_pred, boxes_true)

    # to prevent the dataset division by zero, we add a small value to the denominator
    # and meanwhile filter the result by the intersection check
    iou = filter_iou(intersection_check(boxes_pred, boxes_true), intersection / (union + 1e-7))

    # finally we need to reshape the iou tensor back to (B, N)
    return iou.reshape(B, N)


def test():
    # create a batch of boxes
    boxes_true = torch.tensor([
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4]]).reshape(1, -1, 4)

    boxes_pred = torch.tensor([
        [0.2, 0.2, 0.4, 0.4], # 重合的情况
        [0.0, 0.0, 0.0, 0.0], # 不重合的情况 1
        [0.5, 0.5, 0.0, 0.0], # 不重合的情况 2
        [0.5, 0.5, 0.9, 0.9], # 不重合的情况 3
        [0.0, 0.0, 0.3, 0.3], # 相交的情况 1
        [0.3, 0.3, 0.5, 0.9], # 相交的情况 2
        [0.3, 0.3, 0.3, 0.3], # 相交的情况 3
        [0.0, 0.2, 0.3, 0.9], # 相交的情况 4
        [0.2, 0.3, 0.4, 0.6]]).reshape(1, -1, 4)

    # print out the shape of tensors
    print(boxes_pred.shape)
    print(boxes_true.shape)

    # print out the tensor
    print(boxes_true)
    print(boxes_pred)

    # compute the IoU
    iou = compute_iou_batch(boxes_pred, boxes_true)
    print(iou)


if __name__ == '__main__':
    test()

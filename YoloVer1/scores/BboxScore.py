from YoloVer1.loss.IoU import compute_iou
from YoloVer1.tools.FakeGrid import make_grid
from YoloVer1.tools.TorchSetOp import *

CONFIDENCES = 1
BOUNDING_BOXES = 1


def bbox_score(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    # derive the confidences and ground truth
    pred_conf = output[:, :CONFIDENCES, :]
    true_conf = target[:, :CONFIDENCES, :]

    # derive bounding boxes from output and target
    pred_bboxes = output[:, CONFIDENCES: CONFIDENCES + BOUNDING_BOXES * 4:]
    true_bboxes = target[:, CONFIDENCES: CONFIDENCES + BOUNDING_BOXES * 4:]

    # 求pred_conf和true_conf的交集，并且求交集的iou
    hits_mask = bool_intersection_op(pred_conf > threshold, true_conf > threshold)
    all_iou = compute_iou(pred_bboxes, true_bboxes)

    # 预测成功的iou表
    hits_iou = torch.where(hits_mask, all_iou, torch.zeros_like(all_iou))

    # 计算ious的均值
    average_of_iou = torch.mean(hits_iou[hits_mask])

    # 计算hits数
    hits_num = torch.where(hits_mask, torch.ones_like(hits_mask), torch.zeros_like(hits_mask)).sum()

    # 计算miss数
    miss_mask = bool_difference_op(pred_conf > threshold, true_conf > threshold)
    miss_num = torch.where(miss_mask, torch.ones_like(miss_mask), torch.zeros_like(miss_mask)).sum()

    # 计算error数
    error_mask = bool_difference_op(true_conf > threshold, pred_conf > threshold)
    error_num = torch.where(error_mask, torch.ones_like(error_mask), torch.zeros_like(error_mask)).sum()

    # 计算failure_table
    failure_table = torch.where(error_mask, -1, 0)
    failure_table = torch.where(miss_mask, 1, failure_table)

    # 返回预测成功的bbox数，miss数，error数，以及平均iou, failure_table；
    # failure_table的维度为 [B, N]，数据应该分别为 [0 pass, 1 missed, -1 error]
    return hits_num, miss_num, error_num, average_of_iou, failure_table


def test():
    # 创建一些测试用网格
    test_grid = torch.rand(1, 15, 10)

    # 创建一些测试用数据
    true_0 = make_grid(conf=1, bbox=(0.6, 0.6, 0.8, 0.8), obj=0)
    true_1 = make_grid(conf=1, bbox=(0.6, 0.6, 0.8, 0.8), obj=1)
    true_2 = make_grid(conf=0)
    true_3 = make_grid(conf=0)
    true_4 = make_grid(conf=0)
    true_5 = make_grid(conf=1, bbox=(0.6, 0.6, 0.8, 0.8), obj=5)
    true_6 = make_grid(conf=1, bbox=(0.6, 0.6, 0.8, 0.8), obj=6)
    true_7 = make_grid(conf=1, bbox=(0.6, 0.6, 0.8, 0.8), obj=7)
    true_8 = make_grid(conf=1, bbox=(0.6, 0.6, 0.8, 0.8), obj=8)
    true_9 = make_grid(conf=1, bbox=(0.6, 0.6, 0.8, 0.8), obj=9)

    true_grid = torch.stack([true_0, true_1, true_2, true_3, true_4, true_5, true_6, true_7, true_8, true_9], dim=1)
    true_grid = true_grid.reshape(1, 15, 10)

    hits_num, miss_num, error_num, average_of_iou, failure_table = bbox_score(test_grid, true_grid)
    print(hits_num, miss_num, error_num)
    print(average_of_iou)
    print(failure_table)


if __name__ == '__main__':
    test()

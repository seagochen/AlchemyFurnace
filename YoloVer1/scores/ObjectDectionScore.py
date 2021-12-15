from YoloVer1.tools.FakeGrid import make_grid
from YoloVer1.tools.TorchSetOp import *
from YoloVer1.scores.BboxScore import bbox_score

CONFIDENCES = 1
BOUNDING_BOXES = 1


def object_score(output, target, failure_table):
    # derive the object categories from output and target
    pred_categories = output[:, CONFIDENCES + BOUNDING_BOXES * 4:, :]
    true_categories = target[:, CONFIDENCES + BOUNDING_BOXES * 4:, :]

    # derive the object categories table depending on the failure_table
    pred_categories = bool_mask_select(pred_categories, failure_table == 0, torch.zeros_like(pred_categories))
    true_categories = bool_mask_select(true_categories, failure_table == 0, torch.zeros_like(true_categories))

    # 保留每项中最大值
    _, pred_indices = torch.max(pred_categories, dim=2)
    _, true_indices = torch.max(true_categories, dim=2)

    fin = bool_intersection_op(bool_eq_op(pred_indices, true_indices), failure_table == 0)

    # return the sum of the success object table
    return torch.sum(fin)


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

    print(object_score(test_grid, true_grid, failure_table))


if __name__ == '__main__':
    test()

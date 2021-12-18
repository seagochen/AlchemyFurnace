from YoloVer102.loss.IoU import compute_iou
from YoloVer102.tools.TorchSetOp import *
from YoloVer102.tools.FakeGrid import *


def yolo_scores(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                confidences: int = 1, bounding_boxes: int = 1):
    # keep the dimensions
    B, C, N = output.shape

    # derive the confidences and ground truth
    pred_conf = output[:, :confidences, :]
    true_conf = target[:, :confidences, :]

    # derive bounding boxes from output and target
    pred_bboxes = output[:, confidences: confidences + bounding_boxes * 4, :]
    true_bboxes = target[:, confidences: confidences + bounding_boxes * 4, :]

    # derive the object categories from output and target
    pred_categories = output[:, confidences + bounding_boxes * 4:, :]
    true_categories = target[:, confidences + bounding_boxes * 4:, :]

    # compute iou
    all_iou = compute_iou(pred_bboxes, true_bboxes)

    # derive the conjunction of the confidences of the predicted and ground truth
    # then, filter the iou
    hit_mask = bool_intersection_op(pred_conf > threshold, true_conf > threshold).reshape(B, N)
    hit_iou = torch.where(hit_mask, all_iou, torch.zeros_like(all_iou))

    # calculate average of iou
    hit_sum = torch.sum(hit_iou > 0.)
    sum_iou = torch.sum(hit_iou)

    # calculate average of object accuracy
    hit_pred_classes = torch.where(hit_mask.unsqueeze(1), pred_categories, torch.zeros_like(pred_categories))
    hit_true_classes = torch.where(hit_mask.unsqueeze(1), true_categories, torch.zeros_like(true_categories))

    # select max indices from classes
    max_pred_index = torch.argmax(hit_pred_classes, dim=1)
    max_true_index = torch.argmax(hit_true_classes, dim=1)

    # count how many indices are the same
    hit_correct_classes = torch.eq(max_pred_index, max_true_index).sum()

    # return the hits and average iou
    return hit_sum.item(), sum_iou.item(), hit_correct_classes.item()


if __name__ == "__main__":
    
    # generate grids
    target = torch.rand(4, 15, 64)
    output = torch.rand(4, 15, 64)

    # get the bounding box scores
    yolo_scores(output, target)

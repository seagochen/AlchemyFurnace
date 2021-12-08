import torch
import numpy as np

from yolo.Losses.IntersectionOverUnion import IntersectionOverUnion


def best_bbox(bbox_iou1, bbox_iou2, pre_box1, pre_box2):
    """
    首先对比iou1，iou2，从中按照 每图/每格 挑选中最大的iou，并以此来生成最终的 bbox
    :param bbox_iou1: iou of bbox 1, size of [batch, grids]
    :param bbox_iou2: iou of bbox 2, size of [batch, grids]
    :param pre_box1: bbox 1, size of [batch, grids, (cx, cy, w, h, confidence)]
    :param pre_box2: bbox 2, size of [batch, grids, (cx, cy, w, h, confidence)]
    :return: [batch, (best_iou_of_grids)], [batch, grids, (cx, cy, w, h, confidence)]
    """

    # 记录网格大小
    batch, grids = bbox_iou1.size()

    # 张量变形
    bbox_iou1 = bbox_iou1.reshape(1, -1)
    bbox_iou2 = bbox_iou2.reshape(1, -1)

    # 比较张量大小
    concatenated = torch.cat([bbox_iou1, bbox_iou2], dim=0)
    max_iou, max_indices = torch.max(concatenated, dim=0)

    # 把 bounding box 维度进行调整
    data1_frames = pre_box1.view(-1, 1, 5)  # [batch * grids, 1, (x,y,w,h,c)]
    data2_frames = pre_box2.view(-1, 1, 5)  # [batch * grids, 1, (x,y,w,h,c)]

    # 确定bbox1和bbox2各自的掩码
    shape_of_bb1 = max_indices[...] < 1
    shape_of_bb2 = max_indices[...] >= 1

    # 调整掩码的维度
    shape_of_bb1 = shape_of_bb1.view(-1, 1)  # [batch * grids, 1]
    shape_of_bb2 = shape_of_bb2.view(-1, 1)  # [batch * grids, 1]

    # 分别组合掩码，bbox
    concatenated_mask = torch.cat([shape_of_bb1, shape_of_bb2], dim=1)  # [batch * grids, boxes]
    concatenated_data = torch.cat([data1_frames, data2_frames], dim=1)  # [batch * grids, boxes, (x,y,w,h,c)]

    # 对数据进行过滤，挑选出iou最大项目
    result = concatenated_data[concatenated_mask]  # [batch * grids, (x,y,w,h,c)]
    result = result.view(batch, grids, 5)

    return max_iou.reshape(batch, grids), result


def negative_to_zero(tensor):
    zeros = torch.zeros_like(tensor)

    # 清除掉全部小于0的数，并设置为0
    tensor = torch.where(tensor <= 0, zeros, tensor)

    return tensor


class Preprocess(torch.nn.Module):

    def __init__(self, grids, bbox, categories):
        super().__init__()
        self.grids = grids
        self.bbox = bbox
        self.categories = categories
        self.iou = IntersectionOverUnion()

    def forward(self, predications, targets):
        # reshape all tensors
        predications = predications.reshape(-1, self.grids * self.grids, self.bbox * 5 + self.categories)
        targets = targets.reshape(-1, self.grids * self.grids, self.bbox * 5 + self.categories)

        # fragments
        pre_bbox_1, pre_bbox_2, pre_cats = predications[..., 0:5], predications[..., 5:10], predications[..., 10:]
        tar_bbox_1, tar_bbox_2, tar_cats = targets[..., 0:5], targets[..., 5:10], targets[..., 10:]

        # ious
        iou_of_bbox1 = self.iou(tar_bbox_1[..., :4], pre_bbox_1[..., :4])
        iou_of_bbox2 = self.iou(tar_bbox_2[..., :4], pre_bbox_2[..., :4])

        # 在每张图片每个网格里，找出最好的bbox
        best_iou, predicated_box = best_bbox(iou_of_bbox1, iou_of_bbox2, pre_bbox_1, pre_bbox_2)

        # 从target中仅选取一个 bbox 参与随后的计算
        target_box = tar_bbox_1

        return target_box, tar_cats, predicated_box, pre_cats


class YoLoNetVer1Loss(torch.nn.Module):

    def __init__(self, grids=7, bbox=2, categories=20, lambda_coord=5, lambda_no_obj=.5):
        """
        损失函数计算
        :param grids: 网格大小
        :param bbox: bounding box, yolo v1 论文中每个网格使用两个bbox进行预测
        :param categories: yolo v1 使用的数据集有20种物体类别
        :param lambda_coord: 当网格有物体（即前景）时，就需要计算预测框与标记的IOU，参数用于增加前景在损失函数中的权重
        :param lambda_no_obj: 当网格不包含物体（即背景）时，需要把一部分背景信息加入到损失函数的权重
        """
        super().__init__()
        self.grids = grids
        self.bbox = bbox
        self.categories = categories
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj

        self.mse = torch.nn.MSELoss(reduction="sum")
        self.preprocess = Preprocess(grids, bbox, categories)

    def forward(self, predications, targets):
        """
        在论文中，使用的参数数：
        lambda_no_obj = 0.5
        lambda_coord = 5

        在论文中，YoLo的Loss计算方式是
        loss_of_bbox = lambda_coord * [(X - pre_x)^2 + (Y - pred_y)^2 +
                        (sqrt(W) - sqrt(pre_w))^2 + (sqrt(H) - sqrt(pre_y)^2)]
        loss_of_confidence = (confidence_IoU - pre_ci)^2 + (0 - pre_ci)^2 * lambda_no_obj
        loss_of_classification = sum((type_i - pre_pi)^2, from 0 to 20)

        sum_of_loss = 0
        bounding_boxes = 2

        for i, j in grids.indices():
            for b in bounding_boxes:
                sum_of_loss =
                    loss_of_bbox(bbox(i,j), pre_bbox(i, j, b)) +
                    loss_of_confidence(i, j) +
                    loss_of_classification(i, j)

        return sum_of_loss

        :param predications:
            [batch][7 x 7][x1, y1, w1, h1, confidence1, x2, y2, w2, h2, confidence2, Pr(Class_i | Object)]
        :param targets:
            [batch][7 x 7][x1, y1, w1, h1, confidence1, x2, y2, w2, h2, confidence2, Pr(Class_i | Object)]
        :return: total loss
        """
        ###########################################################################################################
        # 数据预处理阶段
        # target_bbox, size of [batch, grids, (cx, cy, w, h, has_obj)]
        # target_classification, size of [batch, (obj of classes)]
        #
        # predicated_bbox, size of [batch, grids, (cx, cy, w, h, confidence)]
        # predicated_classification, size of [batch, (confidence of classes)]
        ###########################################################################################################
        target_bbox, target_classification, predicated_bbox, predicated_classification = \
            self.preprocess(predications, targets)

        # 获取新的维度信息
        batch, grids, features = target_bbox.size()

        # 包含物体
        contains_obj = target_bbox[..., 4]  # [batch, grids]
        obj_mask = contains_obj.view(batch, grids, 1).expand(batch, grids, features)

        # 不含物体
        no_obj_mask = torch.zeros_like(obj_mask)
        ones = torch.ones_like(obj_mask)
        no_obj_mask = torch.where(obj_mask <= 0, ones, no_obj_mask)

        ###########################################################################################################
        # 计算 loss of bounding box
        #
        # 仅包含物体的 bbox 参与这里的计算
        ###########################################################################################################
        # 过滤不合格的预测
        predicated_valid_box = predicated_bbox * obj_mask

        # 过滤数据
        filtered_predict_box = negative_to_zero(predicated_valid_box)
        filtered_target_bbox = negative_to_zero(target_bbox)

        # 计算 bounding box 损失
        bbox_loss = self.mse(filtered_predict_box[:, 0], filtered_target_bbox[:, 0]) + \
                    self.mse(filtered_predict_box[:, 1], filtered_target_bbox[:, 1]) + \
                    self.mse(torch.sqrt(filtered_predict_box[:, 2]), torch.sqrt(filtered_target_bbox[:, 2])) + \
                    self.mse(torch.sqrt(filtered_predict_box[:, 3]), torch.sqrt(filtered_target_bbox[:, 3]))

        bbox_loss = bbox_loss * self.lambda_coord

        ###########################################################################################################
        #  计算 loss of object confidence
        #
        # 全网格 object confidence 参与这里的计算
        ###########################################################################################################
        # 物体置信度的分类
        predicated_obj = predicated_bbox * obj_mask
        predicated_no_obj = predicated_bbox * no_obj_mask

        # 计算损失
        object_loss = self.mse(predicated_obj[..., 4], target_bbox[..., 4]) + \
                      self.mse(predicated_no_obj[..., 4], target_bbox[..., 4]) * self.lambda_no_obj

        ###########################################################################################################
        # 计算 loss of classification
        #
        # 仅包含物体的网格参与这里的计算
        ###########################################################################################################
        classification_loss = self.mse(predicated_classification, target_classification)

        return bbox_loss + object_loss + classification_loss


def _derive_grid_info(grid_info: np.ndarray):

    max_confidence = 0
    max_i = 0

    for i in range(20):
        confidence = grid_info[10 + i]

        if confidence > max_confidence:
            max_i = i
            max_confidence = confidence

    # 计算第一个bbox的置信度
    score_1 = grid_info[4] * max_confidence

    # 计算第二个bbox的置信度
    score_2 = grid_info[9] * max_confidence

    if score_2 > score_1:
        offset = 5
    else:
        offset = 0

    cent_x = grid_info[offset + 0]
    cent_y = grid_info[offset + 1]
    bbox_w = grid_info[offset + 2]
    bbox_h = grid_info[offset + 3]

    return cent_x, cent_y, bbox_w, bbox_h, max_i


def _has_object(grid_info: np.ndarray):
    return grid_info[4] > .5


def compute_rect(i, j, ground_truth: torch.Tensor):
    from yolo.Dataset.YoLoImageDescription import index_to_object_name

    # 获取当前的bbox信息
    cent_x, cent_y, bbox_w, bbox_h, obj_type = _derive_grid_info(ground_truth[i, j].detach().numpy())

    if obj_type == -1:  # error
        print("error, obj type")
        res = False
    else:
        res = True

    # 先把物体编号转换为名称
    name = index_to_object_name(obj_type)

    # 计算 bounding box 的左上角和长、宽
    width = int(448 * bbox_w)
    height = int(448 * bbox_h)

    # 计算绝对中心点位置
    cx = (cent_x + i) / 7 * 448
    cy = (cent_y + j) / 7 * 448

    # 计算左上角位置
    ltx = int(round(cx - width / 2.))
    lty = int(round(cy - height / 2.))

    return res, name, ltx, lty, width, height


def show_result_with_label(image,
                           predicated: torch.Tensor,
                           ground_truth: torch.Tensor):
    from yolo.Utils.Display import show_object_name_with_rect

    for i in range(7):
        for j in range(7):
            if _has_object(ground_truth[i, j].detach().numpy()):  # 当前网格里有物体

                # predication
                res, name, ltx, lty, width, height = compute_rect(i, j, predicated)

                if res:
                    image = show_object_name_with_rect(image, name,
                                                       (ltx, lty, ltx + width, lty + height),
                                                       (255, 0, 255))
                # ground truth
                res, name, ltx, lty, width, height = compute_rect(i, j, ground_truth)

                if res:
                    image = show_object_name_with_rect(image, name,
                                                       (ltx, lty, ltx + width, lty + height),
                                                       (0, 255, 255))

    return image


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from yolo.Dataset.YoLoDataset import YoLoDataset
    from yolo.Dataset.DownloadDataset import DATA_PATH
    from yolo.Models.YoLoNetVer1 import YoLoNetVer1
    from yolo.Utils.ImgConvertor import tensor_to_cv
    import cv2

    batch_size = 1

    dataset = YoLoDataset(DATA_PATH, distorted=True)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    loss_fn = YoLoNetVer1Loss()
    model = YoLoNetVer1()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    cv2.namedWindow("Samples")

    for idx, (images, labels) in enumerate(train_loader, 1):

        """
         images with size of [batch_size, 3, 448, 448]
         labels with size of [batch_size, 7, 7, 30]
         """

        for epoch in range(500):
            torch.no_grad()

            optimizer.zero_grad()

            """
            predicated values with size of [batch_size, 7, 7, 30]
            """
            predicates = model(images)
            yolo_loss = loss_fn(predicates, labels)

            yolo_loss.backward()
            optimizer.step()

            print(epoch, yolo_loss.item())

            # 展示图片预测结果
            batch, grid1, grid2, features = predicates.size()
            for batch in range(batch):
                img = tensor_to_cv(images[batch])

                # draw rect of predication
                img = show_result_with_label(img, predicates[batch], labels[batch])

                # for obj in img_info.objects:
                cv2.imshow("Samples", img)

                # wait key
                c = cv2.waitKey(1000)
                if c == ord('q'):
                    break
                elif c == ord('n'):
                    continue
        break

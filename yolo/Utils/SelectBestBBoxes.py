import torch


class BestBBox(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, bbox_iou1, bbox_iou2, pre_box1, pre_box2):
        """
        首先对比iou1，iou2，从中按照 每图/每格 挑选中最大的iou，并以此来生成最终的 bbox
        :param bbox_iou1: iou of bbox 1, size of [batch, grids]
        :param bbox_iou2: iou of bbox 2, size of [batch, grids]
        :param pre_box1: bbox 1, size of [batch, grids, (cx, cy, w, h, confidence)]
        :param pre_box2: bbox 2, size of [batch, grids, (cx, cy, w, h, confidence)]
        :return: tensor of [batch, grids, (cx, cy, w, h, confidence)]
        """
        pass


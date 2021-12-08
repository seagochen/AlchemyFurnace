import torch


class GridCoordPtsSort(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dismiss = 0

    def forward(self, x1, x2, x3, x4):
        # just to cease pycharm warning
        self.dismiss = 1

        x1 = x1.view(-1, 1)
        x2 = x2.view(-1, 1)
        x3 = x3.view(-1, 1)
        x4 = x4.view(-1, 1)

        concatenated = torch.cat((x1, x2, x3, x4), dim=1)
        value, idx = torch.sort(concatenated, dim=1)
        return value.transpose(0, 1)  # 对矩阵转置


class IntersectionOverUnion(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.div = 2.
        self.sort = GridCoordPtsSort()

    def forward(self, box1, box2):
        """Implement the intersection over union (IoU) between box1 and box2

        Arguments:
        box1 -- first box, dimensions with (N, grids, (cx, cy, w, h))
        box2 -- second box, dimensions with (N, grids, (cx, cy, w, h))
        """
        # reshape
        batch_size, grids = -1, -1
        if len(box1.size()) > 2:
            batch_size, grids = box1.size()[0],  box1.size()[1]
            box1 = box1.reshape(-1, 4)
            box2 = box2.reshape(-1, 4)

        # compute intersection area
        lx_b1 = box1[..., 0] - box1[..., 2] / self.div
        rx_b1 = box1[..., 0] + box1[..., 2] / self.div
        ty_b1 = box1[..., 1] - box1[..., 3] / self.div
        by_b1 = box1[..., 1] + box1[..., 3] / self.div

        lx_b2 = box2[..., 0] - box2[..., 2] / self.div
        rx_b2 = box2[..., 0] + box2[..., 2] / self.div
        ty_b2 = box2[..., 1] - box2[..., 3] / self.div
        by_b2 = box2[..., 1] + box2[..., 3] / self.div

        x_vals = self.sort(lx_b1, rx_b1, lx_b2, rx_b2)
        y_vals = self.sort(ty_b1, by_b1, ty_b2, by_b2)

        inter_area = (x_vals[2] - x_vals[1]) * (y_vals[2] - y_vals[1])

        # compute union area
        box1_area = box1[..., 2] * box1[..., 3]
        box2_area = box2[..., 2] * box2[..., 3]
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        iou_zeros = torch.zeros_like(iou)
        # 清除掉全部小于0的数，并设置为0
        iou = torch.where(iou <= 0, iou_zeros, iou)

        # return with reshaped
        if batch_size > 0 and grids > 0:
            return iou.reshape(batch_size, grids)
        else:
            return iou



if __name__ == "__main__":
    rect1 = torch.Tensor([[.5, .5, 0.3, 0.3], [.5, .5, 0.3, 0.3], [.5, .5, 0.3, 0.3], [.5, .5, 0.3, 0.3]])
    rect2 = torch.Tensor([[.499, .499, 0.299, 0.299], [.49, .49, 0.4, 0.4], [.49, .49, 0.3, 0.3], [.5, .5, 0.3, 0.3]])

    iou = IntersectionOverUnion()
    print(iou(rect1, rect2))

    rect1 = torch.Tensor([[.5, .5, 0.3, 0.3], [.5, .5, 0.3, 0.3]])
    rect2 = torch.Tensor([[.0, .0, 0., 0.], [.0, .0, 0., 0.]])
    print(iou(rect1, rect2))



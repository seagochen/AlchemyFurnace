import torch


class YoloLoss(torch.nn.Module):

    def __init__(self, confidences=1, bounding_boxes=1, object_categories=10, lambda_coord=5., lambda_noobj=0.5):
        super().__init__()

        # keep parameters
        self.confidences = confidences
        self.bounding_boxes = bounding_boxes
        self.object_categories = object_categories

        # define the lambda
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        # define loss function
        self.loss_function = torch.nn.MSELoss(reduction="mean")

    def forward(self, y_pred, y_true):
        """
        :param y_pred: tensor in shape of [B, C, G]
        :param y_true: tensor in shape of [B, C, G]
        :return:
        """

        # get the shape of dimensions
        B, _, N = y_pred.shape

        # get the confidence and bounding box prediction
        confidence = y_pred[:, :self.confidences, :]
        bounding_box = y_pred[:, self.confidences: self.confidences + self.bounding_boxes * 4, :]
        object_classes = y_pred[:, self.confidences + self.bounding_boxes * 4:, :]

        # get the ground truth
        ground_truth = y_true[:, :self.confidences, :]
        ground_truth_bounding_box = y_true[:, self.confidences: self.confidences + self.bounding_boxes * 4, :]
        ground_truth_object_classes = y_true[:, self.confidences + self.bounding_boxes * 4:, :]

        # lambda
        lambda_coord = torch.where(ground_truth == 1., self.lambda_coord, 1.)
        lambda_noobj = torch.where(ground_truth == 0., self.lambda_noobj, 1.)

        # compute the bounding box loss
        box_loss = self.loss_function(bounding_box, ground_truth_bounding_box) * lambda_coord

        # compute the confidence loss
        conf_loss = self.loss_function(confidence, ground_truth) * lambda_noobj

        # compute the object loss
        object_loss = self.loss_function(object_classes, ground_truth_object_classes) * lambda_noobj

        # compute the total loss
        loss = box_loss + conf_loss + object_loss

        # return the loss
        return torch.sum(loss) / B


"""
论文的方法会导致梯度爆炸，所以在训练过程中，我们直接使用MSE损失函数，更为稳妥一些
"""
import torch


class YoloLoss(torch.nn.Module):

    def __init__(self, grids_size=(1, 1), confidences=0, bounding_boxes=0, object_categories=10):
        super().__init__()

        # keep parameters
        self.grids_size = grids_size
        self.confidences = confidences
        self.bounding_boxes = bounding_boxes
        self.object_categories = object_categories

        # define loss function
        self.loss_function = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: tensor in shape of [B, C, G]
        :param y_true: tensor in shape of [B, C, G]
        :return:
        """

        # get the shape of dimensions
        B_pred, C_pred, G_pred = y_pred.shape
        B_true, C_true, G_true = y_true.shape

        # check if the shape of y_pred and y_true are the same
        assert B_pred == B_true, 'Batch size of y_pred and y_true must be the same'
        assert C_pred == C_true, 'Channel size of y_pred and y_true must be the same'
        assert G_pred == G_true, 'Grid size of y_pred and y_true must be the same'

        # iterate over the grids
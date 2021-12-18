import torch


class YoloV1Network(torch.nn.Module):

    def __init__(self, grids_size=(1, 1), confidences=0, bounding_boxes=0, object_categories=10):
        super().__init__()

        # compute final output dataset size
        self.grids_size = grids_size
        out_features = (confidences + bounding_boxes * 4 + object_categories) * grids_size[0] * grids_size[1]

        # step 1: processing image with convolution layer
        # input dataset size is (B, C: 3, H: 448, W: 448)
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=7 // 2),
            # activate
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output size (B, C: 64, H: 112, W: 112)

        # step 2: processing image with convolution layer
        # input dataset size is (B, C: 64, H: 112, W: 112)
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # activate
            torch.nn.BatchNorm2d(192),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output size (B, C: 192, H: 56, W: 56)

        # step 3: processing image with convolution layer
        # input dataset size is (B, C: 192, H: 56, W: 56)
        self.conv_3 = torch.nn.Sequential(
            # 3.1
            torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            # 3.2
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # 3.3
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            # 3.4
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # activate
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output size (B, C: 512, H: 28, W: 28)

        # step 4: processing image with convolution layer
        # input dataset size is (B, C: 512, H: 28, W: 28)
        self.conv_4 = torch.nn.Sequential(
            # 4.1 - 4.2 x 4
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # 4.3
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            # 4.4
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # activate
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output size (B, C: 1024, H: 14, W: 14)

        # step 5: processing image with convolution layer
        # input dataset size is (B, C: 1024, H: 14, W: 14)
        self.conv_5 = torch.nn.Sequential(
            # 5.1 - 5.2 x 2
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=1 // 2),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # 5.3
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # 5.4
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=3 // 2),
            # activate
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1)
        )  # output size (B, C: 1024, H: 7, W: 7)

        # step 6: processing image with convolution layer
        # input dataset size is (B, C: 1024, H: 7, W: 7)
        self.conv_6 = torch.nn.Sequential(
            # 6.1
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # activate
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1)
        )  # output size (B, C: 1024, H: 7, W: 7)

        # step 7: processing image with linear layer
        # input dataset size is (B, C: 1024 * 7 * 7)
        self.fc_7 = torch.nn.Sequential(
            torch.nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            # activate
            torch.nn.Dropout(),
            torch.nn.LeakyReLU(0.1)
        )  # output size (B, C: 4096)

        # step 8: processing image with linear layer
        # input dataset size is (B, C: 4096)
        self.fc_8 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=out_features),
            # activate
            torch.nn.LeakyReLU(0.1)
        )  # output size (B, C: out_features)

    def forward(self, data):
        # get the dimensions of input dataset
        B, C, H, W = data.shape

        # check input dataset size
        assert C == 3, 'The number of channels of input dataset must be 3.'
        assert H == 448, 'The height of input dataset must be 448.'
        assert W == 448, 'The width of input dataset must be 448.'

        data = self.conv_1(data)
        data = self.conv_2(data)
        data = self.conv_3(data)
        data = self.conv_4(data)
        data = self.conv_5(data)
        data = self.conv_6(data)
        data = self.fc_7(data.reshape(B, -1))
        data = self.fc_8(data)

        # transform the tensor to the shape of [B, C, G]
        data = data.reshape(B, -1, self.grids_size[0] * self.grids_size[1])
        return data


def test_without_params():
    data = torch.zeros(64, 3, 448, 448)
    yolo_v1 = YoloV1Network()
    data = yolo_v1(data)
    print(data.size())


def test_with_params():
    batch_size = 4
    grids_size = (8, 8)
    confidences = 1
    bounding_boxes = 1
    object_categories = 10

    data = torch.zeros(batch_size, 3, 448, 448)
    yolo_v1 = YoloV1Network(grids_size=grids_size,
                            confidences=confidences,
                            bounding_boxes=bounding_boxes,
                            object_categories=object_categories)
    data = yolo_v1(data)
    print(data.size())


if __name__ == "__main__":
    test_without_params()
    test_with_params()

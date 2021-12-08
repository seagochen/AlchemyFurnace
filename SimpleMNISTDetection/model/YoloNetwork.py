import torch


class SimpleYoloNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # step 1: process image (1, 3, 448, 448) with convolution layers
        self.conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=7 // 2),
            # activate
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output size (1, 64, 112, 112)

        # step 2: process tensor (1, 64, 112, 112) with convolution layers
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # activate
            torch.nn.BatchNorm2d(192),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # output size (1, 192, 56, 56)

        # step 3: process tensor (1, 192, 56, 56) with convolution layers
        self.conv_layer3 = torch.nn.Sequential(
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
        )  # output size (1, 512, 28, 28)

        # step 4. process tensor (1, 512, 28, 28) with convolution layers
        self.conv_layer4 = torch.nn.Sequential(
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
        )  # output size (1, 1024, 14, 14)

        # step 5. process tensor (1, 14, 14, 1024) with convolution layers
        self.conv_layer5 = torch.nn.Sequential(
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
        )  # output size (1, 1024, 7, 7)

        # step 6 process tensor (1, 1024, 7, 7) with convolution layers
        self.conv_layer6 = torch.nn.Sequential(
            # 6.1
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2),
            # activate
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1)
        )  # output size (1, 1024, 7, 7)

        # step 7 process flatted tensor (1, 1024, 7, 7) with linear layer
        self.conn_layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            # activate
            torch.nn.Dropout(),
            torch.nn.LeakyReLU(0.1)
        )  # output size (1, 4096)

        # step 8 finally output
        self.conn_layer2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=10),
            # activate
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, data):
        # get the dimensions of input dataset
        B, C, H, W = data.shape

        data = self.conv_layer1(data)
        data = self.conv_layer2(data)
        data = self.conv_layer3(data)
        data = self.conv_layer4(data)
        data = self.conv_layer5(data)
        data = self.conv_layer6(data)
        data = self.conn_layer1(data.reshape(B, -1))
        data = self.conn_layer2(data)
        return data


def test():
    data = torch.zeros(64, 3, 448, 448)
    yolo_v1 = SimpleYoloNetwork()
    data = yolo_v1(data)
    print(data.size())


if __name__ == "__main__":
    test()

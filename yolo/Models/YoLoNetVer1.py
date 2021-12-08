import torch


class YoLoNetVer1(torch.nn.Module):

    def __init__(self, categories=20, bbox=2, debug=False):
        super().__init__()

        # debug message
        self.debug = debug
        self.categories = categories
        self.bnd_boxes = bbox

        # step 1: process image (1, 3, 448, 448) with convolution layers
        # 1.0 COV(64, 7, 7) stride 2
        # 1.N POOL(2, 2) stride 2
        self.conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=7//2),
            # activate
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # output size (1, 64, 112, 112)
        #
        #
        # step 2: process tensor (1, 64, 112, 112) with convolution layers
        # 2.0 COV(192, 3, 3)
        # 2.N POOL(2, 2) stride 2
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # activate
            torch.nn.BatchNorm2d(192),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # output size (1, 192, 56, 56)
        #
        #
        # step 3: process tensor (1, 192, 56, 56) with convolution layers
        # 3.1 COV(128, 1, 1)
        # 3.2 COV(256, 3, 3)
        # 3.3 COV(256, 1, 1)
        # 3.4 COV(512, 3, 3)
        # 3.N POOL(2, 2) stride 2
        self.conv_layer3 = torch.nn.Sequential(
            # 3.1
            torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            # 3.2
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # 3.3
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            # 3.4
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # activate
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # output size (1, 512, 28, 28)
        #
        #
        # step 4. process tensor (1, 512, 28, 28) with convolution layers
        # 4.1 COV(256, 1, 1) -- x4
        # 4.2 COV(512, 3, 3) -- x4
        # 4.3 COV(512, 1, 1)
        # 4.4 COV(1024, 3, 3)
        # 4.N POOL(2, 2) stride 2
        self.conv_layer4 = torch.nn.Sequential(
            # 4.1 - 4.2 x 4
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # 4.3
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            # 4.4
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # activate
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1),
            # max pool
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # output size (1, 1024, 14, 14)
        #
        #
        # step 5. process tensor (1, 14, 14, 1024) with convolution layers
        # 5.1 COV(512, 1, 1)  -- x2
        # 5.2 COV(1024, 3, 3) -- x2
        # 5.3 COV(1024, 3, 3)
        # 5.4 COV(1024, 3, 3), stride 2
        self.conv_layer5 = torch.nn.Sequential(
            # 5.1 - 5.2 x 2
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=1//2),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # 5.3
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # 5.4
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=3//2),
            # activate
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1)
        )
        # output size (1, 1024, 7, 7)
        #
        #
        # step 6 process tensor (1, 1024, 7, 7) with convolution layers
        # 6.1 COV(1024, 3, 3)
        # 6.2 COV(1024, 3, 3)
        self.conv_layer6 = torch.nn.Sequential(
            # 6.1
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=3//2),
            # activate
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1)
        )
        # output size (1, 1024, 7, 7)
        #
        #
        # step 7 process flatted tensor (1, 1024, 7, 7) with linear layer
        self.conn_layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=7*7*1024, out_features=4096),
            # activate
            torch.nn.Dropout(),
            torch.nn.LeakyReLU(0.1)
        )
        # output size (1, 4096)
        #
        #
        # step 8 finally output
        self.conn_layer2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=7*7*(bbox * 5 + categories)),
            # activate
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, data):
        """
        输入一张448 x 448 x 3的彩色图像 然后生成一个 7 x 7 x (2 x 5 + 20) 维度的张量
        :param data: 彩色图像，包含3通道数据，维度大小为 (batch, 3，448, 448)
        :return:
        """
        data = self.conv_layer1(data)
        if self.debug:
            print("Convolution layer output size:", data.size())

        data = self.conv_layer2(data)
        if self.debug:
            print("Convolution layer output size:", data.size())

        data = self.conv_layer3(data)
        if self.debug:
            print("Convolution layer output size:", data.size())

        data = self.conv_layer4(data)
        if self.debug:
            print("Convolution layer output size:", data.size())

        data = self.conv_layer5(data)
        if self.debug:
            print("Convolution layer output size:", data.size())

        data = self.conv_layer6(data)
        if self.debug:
            print("Convolution layer output size:", data.size())

        data = data.view(data.size(0), -1)
        data = self.conn_layer1(data)
        if self.debug:
            print("Linear layer output size:", data.size())

        data = self.conn_layer2(data)
        if self.debug:
            print("Linear layer output size:", data.size())

        """
        根据论文描述，最终生成的张量，其结构是
        [7 x 7 大小的网格][每个网格预测出的 两个可能的Bounding Box（竖直或水平方向，包含坐标：左上角x,y,长宽：w,h）, 
            以及预测的可能有物体的概率 * IOU + 20个类型的各概率值]
        
        简单表述如下：
        [7 x 7][x1, y1, w1, h1, confidence1, x2, y2, w2, h2, confidence2, Pr(Class_i | Object)]
        
        Pr: 预测概率
        Pr(Object): 存在概率，是否存在物体的概率 [0, 1]
        IoU: 预测的矩形框与实际标记的矩形框重叠率
        Pr(Class_i | Object): 条件概率，存在物体时，是类别i的概率
        """
        return data.reshape(-1, 7, 7, (5 * self.bnd_boxes + self.categories))


def test():
    data = torch.zeros(1, 3, 448, 448)
    yolo_v1 = YoLoNetVer1(debug=True)
    data = yolo_v1(data)
    print(data.size())


if __name__ == "__main__":
    test()

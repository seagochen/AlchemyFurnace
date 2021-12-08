import torch


class ConvolutionalNeuralNetwork(torch.nn.Module):

    def __init__(self,
                 grids_cols=1, grids_rows=1,
                 confidences=1, bounding_boxes=1, object_categories=1):
        super().__init__()

        # keep the parameters
        self.grids_cols = grids_cols
        self.grids_rows = grids_rows
        self.confidences = confidences
        self.bounding_boxes = bounding_boxes * 4
        self.object_categories = object_categories

        # define the feature size of output
        features = self.grids_cols * self.grids_rows * (self.confidences + self.bounding_boxes + self.object_categories)

        # define the layers

        # step 1. processing input dataset (B, C:1, H:224, W:224) with a convolutional layer
        self.conv_1 = torch.nn.Sequential(
            # 1.1. convolutional layer
            torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), stride=(1, 1), padding=5 // 2),
            # 1.2. activation function
            torch.nn.LeakyReLU(0.1),
            # 1.3. max pooling layer
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )  # output size (B, C:20, H:112, W:112)

        # step 2. processing input dataset (B, C:20, H:112, W:112) with a convolutional layer
        self.conv_2 = torch.nn.Sequential(
            # 2.1. convolutional layer
            torch.nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), stride=(1, 1), padding=5 // 2),
            # 2.2. activation function
            torch.nn.LeakyReLU(0.1),
            # 2.3. max pooling layer
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )  # output size (B, C:50, H:56, W:56)

        # step 3. processing input dataset (B, C:50, H:56, W:56) with a convolutional layer
        self.conv_3 = torch.nn.Sequential(
            # 3.1. convolutional layer
            torch.nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(5, 5), stride=(1, 1), padding=5 // 2),
            # 3.2. activation function
            torch.nn.LeakyReLU(0.1),
            # 3.3. max pooling layer
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )  # output size (B, C:100, H:28, W:28)

        # step 4. processing input dataset (B, C:100, H:28, W:28) with a fully connected layer
        self.fc_4 = torch.nn.Sequential(
            # 4.1. fully connected layer
            torch.nn.Linear(in_features=100 * 28 * 28, out_features=5120),
            # 4.2. activation function
            torch.nn.LeakyReLU(0.1),
            # 4.3. dropout layer
            torch.nn.Dropout(p=0.5)
        )  # output size (B, C:5120)

        # step 5. processing input dataset (B, C:5120) with a fully connected layer
        self.fc_5 = torch.nn.Sequential(
            # 5.1. fully connected layer
            torch.nn.Linear(in_features=5120, out_features=2048),
            # 5.2. activation function
            torch.nn.LeakyReLU(0.1),
            # 5.3. dropout layer
            torch.nn.Dropout(p=0.5)
        )  # output size (B, C:2048)

        # step 6. processing input dataset (B, C:2048) with a fully connected layer
        self.fc_6 = torch.nn.Linear(in_features=2048, out_features=1024)

        # step 7. processing input dataset (B, C:1024) with a fully connected layer
        self.fc_7 = torch.nn.Linear(in_features=1024, out_features=features)

        # step 8. softmax layer
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data):
        # get the dimensions of input dataset
        B, C, H, W = data.shape

        # step 1. processing input dataset (B, C:1, H:224, W:224) with a convolutional layer
        data = self.conv_1(data)
        # step 2. processing input dataset (B, C:20, H:112, W:112) with a convolutional layer
        data = self.conv_2(data)
        # step 3. processing input dataset (B, C:50, H:56, W:56) with a convolutional layer
        data = self.conv_3(data)
        # step 4. processing input dataset (B, C:100, H:28, W:28) with a fully connected layer
        data = self.fc_4(data.view(B, -1))
        # step 5. processing input dataset (B, C:2048) with a fully connected layer
        data = self.fc_5(data)
        # step 6. processing input dataset (B, C:1024) with a fully connected layer
        data = self.fc_6(data)
        # step 7. processing input dataset (B, C:1024) with a fully connected layer
        data = self.fc_7(data)
        # step 8. softmax layer
        data = self.softmax(data)

        return data


def test():
    # define the input dataset
    mat = torch.randn(64, 1, 28, 28)

    # define the network
    network = ConvolutionalNeuralNetwork(grids_cols=5, grids_rows=5, confidences=1, bounding_boxes=2,
                                         object_categories=10)

    # get the output
    output = network(mat)

    # print out the output
    print(output.shape)


if __name__ == "__main__":
    test()

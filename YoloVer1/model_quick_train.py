import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from YoloVer1.dataset.MNISTDataset import MNISTDataset

# global variables
batch_size = 4
epochs = 10
grids_size = (6, 6)
confidences = 1
bounding_boxes = 2
object_categories = 10
data_dir = '../data/MNIST'

# transform sequential
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])

# training dataset
train_dataset = MNISTDataset(root=data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform)

# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)


# define training function
def train(model, data, target, optimizer):
    # train parameters
    model.train()  # set model to train mode

    # criterion and device auto-chosen
    criterion = nn.CrossEntropyLoss()

    # clear the gradients
    optimizer.zero_grad()

    # forward, backward, update
    outputs = model(data)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    print('Train Loss {}'.format(loss.item()))


def run_train_and_test_demo():
    # import model
    from SimpleMNISTDetection.model.YoloNetwork import SimpleYoloNetwork

    # define model
    model = SimpleYoloNetwork()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # get dataset and target
    for batch_idx, (data, target) in enumerate(train_loader):

        # train model in simple way
        for i in range(10):
            train(model, data, target, optimizer)

        # print out debug message
        print("Training with 10 times done!")

    # save model
    print('All finished!')


if __name__ == '__main__':
    run_train_and_test_demo()

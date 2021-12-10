import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from YoloVer1.model.YoloNetwork import YoloV1Network
from YoloVer1.dataset.MNISTDataset import MNISTDataset

# global variables
batch_size = 4
epochs = 10
grids_size = (6, 6)
confidences = 1
bounding_boxes = 2
object_categories = 10
data_dir = 'data/MNIST'

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

# test dataset
test_dataset = MNISTDataset(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform)
# test loader
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# define training function
def train(model, device, loader, optimizer, epoch):
    # train parameters
    model.train()  # set model to train mode

    # criterion and device auto-chosen
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    # train the model
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # clear the gradients
        optimizer.zero_grad()

        # forward, backward, update
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                       100. * batch_idx / len(loader), loss.item()))


# define test function
def test(model, device, loader):
    # test parameters
    model.eval()  # set model to test mode
    test_loss = 0
    correct = 0

    # criterion and device auto-chosen
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    # test the model
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicated = torch.max(output.data, dim=1)
            correct += (predicated == target).sum().item()
            test_loss += criterion(output, target).item()

    test_loss /= len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))


def run_train_and_test_demo():
    # define model
    model = YoloV1Network(grids_size=grids_size,
                          confidences=confidences,
                          bounding_boxes=bounding_boxes,
                          categories=object_categories)

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train model
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # save model
    torch.save(model.state_dict(), 'model/saved_model.pt')
    print('Model saved!')
    print('Done!')


if __name__ == '__main__':
    run_train_and_test_demo()

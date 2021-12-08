import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from SimpleMNISTDetection.dataset.ModifiedDataset import MNISTWrapperDataset

# global variables
batch_size = 4
epochs = 10
grids_cols = 1
grids_rows = 1
confidences = 0
bounding_boxes = 0
object_categories = 10
data_dir = '../data/MNIST'

# transform sequential
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])

# training dataset
train_dataset = MNISTWrapperDataset(root=data_dir,
                               train=True,
                               download=True,
                               transform=transform)
# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

# test dataset
test_dataset = MNISTWrapperDataset(root=data_dir,
                              train=False,
                              download=True,
                              transform=transform)
# test loader
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# define training function
def train(model, device, train_loader, optimizer, epoch):

    # train parameters
    model.train()  # set model to train mode

    # criterion and device auto-chosen
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    # train the model
    for batch_idx, (data, target) in enumerate(train_loader):
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        

# define test function
def test(model, device, test_loader):

    # test parameters
    model.eval()  # set model to test mode
    test_loss = 0
    correct = 0

    # criterion and device auto-chosen
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    # test the model
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicated = torch.max(output.data, dim=1)
            correct += (predicated == target).sum().item()
            test_loss += criterion(output, target).item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def run_train_and_test_demo():
    # import model
    # from SimpleMNISTDetection.model.CNNNetwork import ConvolutionalNeuralNetwork
    from SimpleMNISTDetection.model.YoloNetwork import SimpleYoloNetwork

    # define model
    # model = SimpleYoloNetwork(grids_cols, grids_rows, confidences, bounding_boxes, object_categories)
    model = SimpleYoloNetwork()

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

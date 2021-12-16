import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from YoloVer1.dataset.MNISTDataset import MNISTDataset, GenerateRandMNIST
from YoloVer1.grids.YoloGrids import YoloGrids
from YoloVer1.model.YoloNetwork import YoloV1Network
from YoloVer1.tools.Normalizer import *
from YoloVer1.scores.BboxScore import bbox_score
from YoloVer1.scores.ObjectDectionScore import object_score


# global variables
epochs = 10
batch_size = 4
grids_size = (8, 8)
confidences = 1
bounding_boxes = 1
object_categories = 10

# data folder
data_dir = 'YoloVer1/data/MNIST'

# model folder
model_dir = 'YoloVer1/model/'

# training dataset
train_dataset = MNISTDataset(root=data_dir, train=True, download=True,
                             rand_mnist=GenerateRandMNIST(),
                             grids_system=YoloGrids(),
                             norm_data=generic_normalize)

# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

# test dataset
test_dataset = MNISTDataset(root=data_dir, train=False, download=True,
                            rand_mnist=GenerateRandMNIST(),
                            grids_system=YoloGrids(),
                            norm_data=generic_normalize)

# test loader
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# define training function
def train(model, device, loader, optimizer, epoch):
    # train parameters
    model.train()  # set model to train mode

    # criterion and device auto-chosen
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # train the model
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # clear the gradients
        optimizer.zero_grad()

        # forward, backward, update
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()))


# define test function
def test(model, device, loader):

    # set model to test mode
    model.eval()

    # define some parameters
    correct_of_bbox = 0
    missing_of_bbox = 0
    failed_of_bbox = 0
    correct_of_class = 0
    average_of_iou = 0.

    # device auto-chosen
    model = model.to(device)

    # test the model
    with torch.no_grad():

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # forward only
            output = model(data)

            # test the model
            hits_num, misses_num, fails_num, iou_average, failure_table = bbox_score(output, target)
            object_num = object_score(output, target, failure_table)

            # 统计计算结果
            correct_of_bbox += hits_num
            missing_of_bbox += misses_num
            failed_of_bbox += fails_num
            correct_of_class += object_num
            average_of_iou += iou_average

        # 打印本论测试结果，一共预测成功多少个bbox，猜错多少个，miss多少个，成功的bbox下，预测object的正确率是多少，以及平均iou情况
        accuracy_of_bbox = correct_of_bbox / len(loader.dataset)
        accuracy_of_class = correct_of_class / len(loader.dataset)

        print("\nTest set: Average of bbox accuracy: {:.4f}, average of object accuracy: {:.4f}, average of iou: {:.4f}".format(
            accuracy_of_bbox, accuracy_of_class, average_of_iou / len(loader.dataset)))
        print("Bounding boxes status: hit: {}, miss: {}, failed: {}, total: {}".format(
            correct_of_bbox, missing_of_bbox, failed_of_bbox, len(loader.dataset)))


def run_train_and_test_demo():
    # define model
    model = YoloV1Network(grids_size=grids_size,
                          confidences=confidences,
                          bounding_boxes=bounding_boxes,
                          object_categories=object_categories)

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train model
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # save model
    torch.save(model.state_dict(), '{}/saved_model.pt'.format(model_dir))
    print('Model saved!')
    print('Done!')


if __name__ == '__main__':
    run_train_and_test_demo()

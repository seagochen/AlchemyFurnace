import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from YoloVer100.model.YoloNetwork import YoloV1Network
from Generic.dataset.MNIST.MNISTDataset import MNISTDataset, GenerateRandMNIST
from Generic.tools.Normalizer import generic_normalize
from Generic.grids.YoloGrids import YoloGrids
from Generic.scores.YoloScores import yolo_scores

# global variables
epochs = 10
batch_size = 4
grids_size = (8, 8)
confidences = 1
bounding_boxes = 1
object_categories = 10

# data folder
data_dir = 'data/MNIST'

# model folder
model_dir = 'YoloVer100/model'

# training dataset
train_dataset = MNISTDataset(root=data_dir, train=True, download=True,
                             rand_mnist=GenerateRandMNIST(),
                             grids_system=YoloGrids(),
                             norm_data=generic_normalize)

# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)


# define training function
def train(model, data, target, criterion, optimizer, epoch):
    # train parameters
    model.train()  # set model to train mode

    # clear the gradients
    optimizer.zero_grad()

    # forward, backward, update
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print('Train Epoch: {} loss: {: .6f}'.format(epoch, loss.item()))


# define test function
def test(model, data, target):

    # set model to test mode
    model.eval()

    # define some parameters
    correct_of_bbox = 0
    correct_of_class = 0
    average_of_iou = 0.

    # test the model
    with torch.no_grad():

        # forward only
        output = model(data)

        # test the model
        hits_num, sum_iou, sum_classes = yolo_scores(output, target)

        # 统计计算结果
        correct_of_bbox += hits_num
        average_of_iou += sum_iou
        correct_of_class += sum_classes

        # statistic the predication level
        accuracy_of_iou = average_of_iou / correct_of_class
        accuracy_of_bbox = correct_of_bbox
        accuracy_of_class = correct_of_class

        print("\nTest set: Average of bbox accuracy: {:.4f}".format(accuracy_of_bbox),
              "average of bounding box accuracy: {:.4f}".format(accuracy_of_iou),
              "average of object accuracy: {:.4f}".format(accuracy_of_class))


def run_quick_verification():
    # define model
    model = YoloV1Network(grids_size=grids_size,
                          confidences=confidences,
                          bounding_boxes=bounding_boxes,
                          object_categories=object_categories)

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # criterion and device auto-chosen
    criterion = nn.MSELoss().to(device)

    # load data and target
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # load model to the proper device
        model = model.to(device)

        # train and test the model
        for epoch in range(1, epochs + 1):
            train(model, data, target, criterion, optimizer, epoch)
            test(model, data, target)

        break

    # save model
    torch.save(model.state_dict(), '{}/qv_model.pt'.format(model_dir))
    print('Model saved!')
    print('Done!')


if __name__ == '__main__':
    run_quick_verification()

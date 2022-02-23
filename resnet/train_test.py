import os

from custom_dataset import PvImageFeatureDataset
from custom_dataset_3channel import PvImage3ChannelFeatureDataset

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import math
from custom_res50 import resnet50
from custom_resnet101 import resnet101

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCH = 200


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),

    ])
    # transform_normalize = transforms.Normalize([6.3861e-05, 3.1319e-01, 3.3415e-01, 3.0482e-01],
    #                                            [4.9268e-04, 4.6379e-01, 1.5182e+00, 9.3921e-01])
    # transform_normalize = transforms.Normalize([0, 0, 3.3415e-01, 3.0482e-01],
    #                                            [1, 1, 1.5182e+00, 9.3921e-01])
    transform_normalize = transforms.Normalize([6.3932e-05, 4.9000e-02, 8.3891e-02],
                                               [4.9296e-04, 9.4700e-01, 5.7754e-01])

    index_dir = "../index/combined_ordinary_cliff_10min.csv"
    ordinary = "../extracted_10min_before_cliff"
    cliff = "../extracted_cliff"

    scratch_dir = "/scratch/itee/uqsxu13/nowcasting_feature_data"
    index_dir_linux = os.path.join(scratch_dir, "index/combined_ordinary_cliff_10min.csv")
    ordinary_linux = os.path.join(scratch_dir, "extracted_10min_before_cliff")
    cliff_linux = os.path.join(scratch_dir, "extracted_cliff")

    if os.name != "nt":
        index_dir = index_dir_linux
        ordinary = ordinary_linux
        cliff = cliff_linux

    dataset = PvImage3ChannelFeatureDataset(index_dir, ordinary, cliff, transform=transform,
                                            transform_extra=transform_normalize)

    all_data_size = len(dataset)
    train_size = int(0.8 * all_data_size)
    test_size = all_data_size - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    net = resnet50()
    # net = resnet101()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # The training module
    if not os.path.exists("resnet50.pth"):
        for epoch in range(EPOCH):
            for i, data in enumerate(train_loader, 1):
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                inputs = inputs.float()

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                # print(outputs, i, loss.item())
                # if i == 5:
                #     raise Exception()
                loss.backward()
                optimizer.step()

            print("Epoch: {} / {}".format(epoch + 1, EPOCH))
            print("Current loss: ", loss.item())

        torch.save(net.state_dict(), "resnet50.pth")
        print("Finish training!!!")
    else:
        print("Existed NN")
        net.load_state_dict(torch.load("resnet50.pth", map_location=device))
        net.eval()

    print("Device is:", device)
    total = 0
    correct = 0
    train_total = 0
    train_correct = 0

    # predict_result = torch.tensor([]).to(device)

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()
            outputs = net(inputs)
            predict = (outputs > 0).type(torch.uint8).squeeze()
            total += labels.size(0)
            correct += (predict == labels).sum().item()

            # predict_result = torch.cat((predict_result, predict))
            # print("predicted:", predict)
            # print("labels:", labels)

        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()
            outputs = net(inputs)
            predict = (outputs > 0).type(torch.uint8).squeeze()
            train_total += labels.size(0)
            train_correct += (predict == labels).sum().item()

    print("Training accuracy: ", 100 * train_correct / train_total)
    print("Test accuracy: ", 100 * correct / total)

    # torch.save(predict_result, "predict_result.pt")


if __name__ == "__main__":
    main()

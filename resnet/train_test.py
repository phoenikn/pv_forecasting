import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from custom_dataset import PvImageFeatureDataset
from custom_dataset_3channel import PvImage3ChannelFeatureDataset
from custom_dataset_4channel import PvImage4ChannelFeatureDataset
from custom_dataset_3image_8stack import PvImage8ChannelFeatureDataset
from custom_dataset_two_grayscale import PvImage2ChannelFeatureDataset
from custom_dataset_two_grayscale_numeric import PvImage2ChannelNumericDataset
from custom_dataset_3image_8stack_numeric import PvImage8ChannelNumericDataset

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import math
from custom_res50 import resnet50
from custom_resnet101 import resnet101

BATCH_SIZE = 64
LEARNING_RATE = 0.001
INPUT_TENSOR_AMOUNT = 8
EPOCH = 100


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),

    ])
    # transform_normalize = transforms.Normalize([6.3932e-05, 4.9000e-02, 8.3891e-02],  # without mean the velocities
    #                                            [4.9296e-04, 9.4700e-01, 5.7754e-01])
    # transform_normalize = transforms.Normalize([6.3932e-05, 4.8070e-02, 7.8432e-02],  # With mean the velocities
    #                                            [4.9296e-04, 7.7191e-01, 4.3708e-01])
    # transform_normalize =\
    #     transforms.Normalize([6.3932e-05, 2.6518e-02, 7.3403e-04, 1.3235e-01], # without mean the velocities
    #                                            [0.0005, 0.4769, 0.2787, 0.1106])  # and with gray images
    transform_normalize = \
        transforms.Normalize([6.3935e-05, 2.5340e-02, -1.1304e-03, 1.3301e-01, 6.3839e-05,
                              -1.7348e-02, -2.3021e-02, 1.3414e-01],  # three images, 8stack
                             [4.9297e-04, 4.8047e-01, 2.7523e-01, 1.1130e-01, 4.9260e-04, 5.2269e-01,
                              4.1887e-01, 1.1561e-01])
    # transform_normalize =\
    #     transforms.Normalize([1.3301e-01, 1.3414e-01], [1.1130e-01, 1.1561e-01])  # only two gray images

    index_dir = "../index/combined_ordinary_cliff_10min_three.csv"
    index_numeric_dir = "../index/cliff_time_points_three_numeric.csv"
    ordinary = "../extracted_10min_before_cliff"
    cliff = "../extracted_cliff"
    ordinary_with_gray = "../extracted_10min_before_cliff_with_gray"
    cliff_with_gray = "../extracted_cliff_with_gray"
    ordinary_three = "../extracted_10min_three"
    cliff_three = "../extracted_cliff_three"

    scratch_dir = "/scratch/itee/uqsxu13/nowcasting_feature_data"
    index_dir_linux = os.path.join(scratch_dir, "index/combined_ordinary_cliff_10min_three.csv")
    index_numeric_dir_linux = os.path.join(scratch_dir, "index/cliff_time_points_three_numeric.csv")
    ordinary_linux = os.path.join(scratch_dir, "extracted_10min_before_cliff")
    cliff_linux = os.path.join(scratch_dir, "extracted_cliff")
    ordinary_linux_with_gray = os.path.join(scratch_dir, "extracted_10min_before_cliff_with_gray")
    cliff_linux_with_gray = os.path.join(scratch_dir, "extracted_cliff_with_gray")
    ordinary_linux_three = os.path.join(scratch_dir, "extracted_10min_three")
    cliff_linux_three = os.path.join(scratch_dir, "extracted_cliff_three")

    if os.name != "nt":
        index_dir = index_dir_linux
        index_numeric_dir = index_numeric_dir_linux
        ordinary = ordinary_linux
        cliff = cliff_linux
        ordinary_with_gray = ordinary_linux_with_gray
        cliff_with_gray = cliff_linux_with_gray
        ordinary_three = ordinary_linux_three
        cliff_three = cliff_linux_three

    dataset = PvImage8ChannelNumericDataset(index_numeric_dir, ordinary_three, cliff_three, transform=transform,
                                            transform_extra=transform_normalize)

    all_data_size = len(dataset)
    train_size = int(0.6 * all_data_size)
    validation_size = int(0.2 * all_data_size)
    test_size = all_data_size - train_size - validation_size

    train_set, validation_set, test_set = torch.utils.data.random_split(dataset,
                                                                        [train_size, validation_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    net = resnet50(INPUT_TENSOR_AMOUNT)
    # net = resnet101()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # The training module
    if not os.path.exists("resnet50.pth"):

        records = pd.DataFrame(data={"epoch": [], "train_loss": [], "val_loss": []})
        print("Start training!")

        for epoch in range(EPOCH):

            net.train()
            running_train_loss = 0.0
            for i, data in enumerate(train_loader, 1):
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                inputs = inputs.float()

                optimizer.zero_grad()

                outputs = net(inputs)
                train_loss = criterion(outputs, labels.unsqueeze(1))
                running_train_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()
            avg_train_loss = running_train_loss / i

            net.eval()
            running_val_loss = 0.0
            for i, data in enumerate(validation_loader, 1):
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                inputs = inputs.float()
                outputs = net(inputs)
                val_loss = criterion(outputs, labels.unsqueeze(1))
                running_val_loss += val_loss.item()
            avg_val_loss = running_val_loss / i

            print("Epoch: {} / {}".format(epoch + 1, EPOCH))
            print("Average train loss: ", avg_train_loss)
            print("Average validation loss: ", avg_val_loss)

            records = records.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss},
                                     ignore_index=True)

        records.to_csv("training_records_{}epochs_8stack_numeric.csv".format(EPOCH), index=False)
        torch.save(net.state_dict(), "resnet50.pth")
        print("Finish training!!!")
    else:
        print("Existed NN")
        net.load_state_dict(torch.load("resnet50.pth", map_location=device))
        net.eval()

    print("Device is:", device)

    with torch.no_grad():
        val_model_numeric(net, test_loader, device)

    # predict_result = torch.tensor([]).to(device)

    # with torch.no_grad():
    #     train_accuracy = val_model_on_test_set(net, train_loader, device)
    #     test_accuracy = val_model_on_test_set(net, test_loader, device)
    #
    # print("Training accuracy: ", train_accuracy)
    # print("Test accuracy: ", test_accuracy)

    # torch.save(predict_result, "predict_result.pt")


def val_model_numeric(model, data_loader, device):
    x = range(BATCH_SIZE)
    for data in data_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.float()
        outputs = model(inputs)
        plt.plot(x, np.absolute(outputs.squeeze().numpy() - labels.numpy()) * (-1), x, labels.numpy())
        plt.legend(["difference", "real data"])
        plt.show()
        # print(outputs.squeeze())
        # print(labels)
        break


def val_model_on_test_set(model, data_loader, device):
    total = 0
    correct = 0

    for data in data_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.float()
        outputs = model(inputs)
        predict = (outputs > 0).type(torch.uint8).squeeze()
        total += labels.size(0)
        correct += (predict == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    main()

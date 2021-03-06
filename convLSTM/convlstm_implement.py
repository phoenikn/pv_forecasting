import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import convlstm as cl
from nn_tool.custom_dataset import PvImageFeatureDataset
from nn_tool import mean_std_calculator
from nn_tool.path_by_os import get_path

BATCH_SIZE = 64
LEARNING_RATE = 0.001
INPUT_TENSOR_AMOUNT = 8
EPOCH = 150
INPUT_LEN = 6

# INDEX_FOLDER = "../index"

DATASET_ARG = {
    "index_path": get_path("test.csv"),
    "file_folder": get_path("test"),
    "channels": 6,
    "tensor_size": (364, 364),
    "select_feature": None
}

LSTM_ARG = {
    "input_dim": 1,
    "hidden_dim": [4, 4],
    "kernel_size": (3, 3),
    "num_layers": 2,
    "batch_first": True,
    "bias": True,
    "return_all_layers": False
}


def implement():
    model = cl.ConvLSTMOut(INPUT_LEN, LSTM_ARG)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = PvImageFeatureDataset(**DATASET_ARG, transform=transform)

    mean, std = mean_std_calculator.calculate(dataset)

    transform_norm = transforms.Normalize(mean, std)

    dataset = PvImageFeatureDataset(**DATASET_ARG, transform=transform, transform_extra=transform_norm)

    all_data_size = len(dataset)
    train_size = int(0.6 * all_data_size)
    validation_size = int(0.2 * all_data_size)
    test_size = all_data_size - train_size - validation_size

    train_set, validation_set, test_set = torch.utils.data.random_split(dataset,
                                                                        [train_size, validation_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists("convlstm.pth"):

        records = pd.DataFrame(data={"epoch": [], "train_loss": [], "val_loss": []})
        print("Start training!")

        for epoch in range(EPOCH):

            model.train()
            running_train_loss = 0.0
            for i, data in enumerate(train_loader, 1):
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                inputs = inputs.float()

                optimizer.zero_grad()

                outputs = model(inputs)
                train_loss = criterion(outputs, labels.unsqueeze(1))
                running_train_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()
            avg_train_loss = running_train_loss / i

            model.eval()
            running_val_loss = 0.0
            for i, data in enumerate(validation_loader, 1):
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()
                inputs = inputs.float()
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.unsqueeze(1))
                running_val_loss += val_loss.item()
            avg_val_loss = running_val_loss / i

            print("Epoch: {} / {}".format(epoch + 1, EPOCH))
            print("Average train loss: ", avg_train_loss)
            print("Average validation loss: ", avg_val_loss)

            records = records.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss},
                                     ignore_index=True)

        records.to_csv("training_records_{}epochs_8stack_numeric.csv".format(EPOCH), index=False)
        torch.save(model.state_dict(), "convlstm.pth")
        print("Finish!")
    else:
        print("There is existed model, start validation.")
        model.load_state_dict(torch.load("convlstm.pth", map_location=device))
        model.eval()

    print("Device is:", device)

    with torch.no_grad():
        if os.name == "nt":
            val_model_numeric(model, test_loader, device)


def val_model_numeric(model, data_loader, device):
    x = range(BATCH_SIZE)
    ape_total = 0
    for i, data in enumerate(data_loader, 1):
        start = time.time()
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.float()
        outputs = model(inputs)

        ape_total += ape_calculation(outputs.squeeze(), labels)

        if i == 1:
            print("Spend time for 16 prediction: ", time.time() - start)
            plt.plot(x, outputs.squeeze().numpy(), x, labels.numpy())
            plt.legend(["pred", "real data"])
            plt.title("Prediction of the first batch")
            plt.show()

        # print(outputs.squeeze())
        # print(labels)

    print("MAPE of the test set is:", (ape_total / len(data_loader.dataset)) * 100, "%")


def ape_calculation(pred, actual):
    pred, actual = np.array(pred), np.array(actual)
    return np.sum(np.abs((actual - pred) / actual))


if __name__ == "__main__":
    implement()

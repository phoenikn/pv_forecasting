import os
import sys
import time
import datetime
from os.path import join as path_join
from convLSTM.convlstm import ConvLSTMOut

import pandas as pd
from einops import rearrange

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import torchvision

from nn_tool.raw_img_dataset import RawSkyImageDataset


if os.name == "nt":
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
    ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
    ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"
    EXOGENOUS_DIR = "C:/Users/s4544852/Desktop/gatton PV data/exogenous"
else:
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "/scratch/itee/uqsxu13/2020_data/pre_compress"
    ABSOLUTE_INDEX_DIR = "/scratch/itee/uqsxu13/2020_data/2020_index"
    EXOGENOUS_DIR = "/scratch/itee/uqsxu13/2020_data/exogenous"

BATCH_SIZE = 5 if os.name == "nt" else 8
LEARNING_RATE = 0.0001 if os.name == "nt" else 0.00001
EPOCH = 10
TRAIN_AGAIN = False
NUM_MIN = 5

DATASET_ARG = {
    "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "small": False,
    "output_length": 1,
    "mode": "",
    "input_length": NUM_MIN,
    "interval": 20,
    # the real interval + 5
    "tensor_size": (256, 256),
    "exogenous_folder": EXOGENOUS_DIR,
    "transform": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1119, 0.1167, 0.1461),
                             (0.116, 0.1160, 0.1288))
    ])
}

CONVLSTM_MODEL_PATH = "convlstm_{}_interval.pth".format(DATASET_ARG["interval"]-5)

def forward_model(device, data, optimizer, model, criterion, is_train=False):
    inputs, labels, _, _ = [element.to(device) for element in data]
    labels = labels.float()
    inputs = inputs.float()
    print(inputs.size())
    # inputs = rearrange(inputs, "b i c h w -> b (i c) h w")

    if is_train:
        optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if is_train:
        loss.backward()
        optimizer.step()
    if os.name == "nt":
        print(loss.item())
        print(torch.flatten(outputs.data))
        print(torch.flatten(labels))
        print("------------" * 5)
    return loss


def implement():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------Dataset setting-----------------------------------
    dataset = RawSkyImageDataset(**DATASET_ARG)
    all_data_size = len(dataset)
    train_size = int(0.6 * all_data_size)
    validation_size = int(0.2 * all_data_size)
    test_size = all_data_size - train_size - validation_size
    train_set, validation_set, _ = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

    # --------------------------------Model initialize-----------------------------------
    model = ConvLSTMOut(5, {
        "input_dim": 3,
        "hidden_dim": [4, 4],
        "kernel_size": (3, 3),
        "num_layers": 2,
        "batch_first": True,
        "bias": True,
        "return_all_layers": False
    })
    if TRAIN_AGAIN:
        model.load_state_dict(torch.load(CONVLSTM_MODEL_PATH, map_location=device))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    if (not os.path.exists(CONVLSTM_MODEL_PATH)) or (os.path.exists(CONVLSTM_MODEL_PATH) and TRAIN_AGAIN):

        records = pd.DataFrame(data={"epoch": [], "train_loss": [], "val_loss": []})
        print("Start training!")
        last_time = datetime.datetime.now()
        sys.stdout.flush()

        best_loss = 100

        for epoch in range(EPOCH):

            model.train()
            running_train_loss = 0.0
            ten_ratio = -1
            for i, data in enumerate(train_loader, 1):
                # -------------------ratio print------------------------------
                ratio = int(i / len(train_loader) * 10)
                if ratio != ten_ratio:
                    print("{}%".format(ratio * 10))
                    sys.stdout.flush()
                    ten_ratio = ratio

                # ------------------train--------------------------------------
                train_loss = forward_model(device=device, data=data, optimizer=optimizer,
                                           model=model, criterion=criterion, is_train=True)
                running_train_loss += train_loss.item()
            avg_train_loss = running_train_loss / i

            print("start validation")
            sys.stdout.flush()

            model.eval()
            running_val_loss = 0.0
            for i, data in enumerate(validation_loader, 1):
                val_loss = forward_model(device=device, data=data, optimizer=optimizer,
                                         model=model, criterion=criterion, is_train=False)
                running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / i

            print("Epoch: {} / {}".format(epoch + 1, EPOCH))
            print("Average train loss: ", avg_train_loss)
            print("Average validation loss: ", avg_val_loss)
            now_time = datetime.datetime.now()
            time_usage = now_time - last_time
            print("Finish time: ", now_time)
            print("Time usage: ", time_usage)
            last_time = now_time
            print("Estimate next epoch finish: ", now_time + time_usage)
            sys.stdout.flush()

            if avg_val_loss < best_loss:
                torch.save(model.state_dict(), "{}_best_interval{}.pth".format("CONVLSTM", DATASET_ARG["interval"]-5))
                best_loss = avg_val_loss

            records = records.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss},
                                     ignore_index=True)

        records.to_csv("training_records_{}epochs_CONVLSTM_loss.csv".format(EPOCH), index=False)
        torch.save(model.state_dict(), CONVLSTM_MODEL_PATH)
        print("Finish!")
    else:
        print("There is existed model, start validation.")

    print("Device is:", device)


if __name__ == "__main__":
    torch.manual_seed(0)
    implement()

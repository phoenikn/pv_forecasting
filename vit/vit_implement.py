import datetime
import functools
import os
import sys
import time
from os.path import join as path_join
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

from vit import ViT, TimeVIT
from historical_mix_model import MixHistorical
from nn_tool.raw_img_dataset import RawSkyImageDataset
from nn_tool.path_by_os import get_path

if os.name == "nt":
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
    ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
    ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"
    EXOGENOUS_DIR = "C:/Users/s4544852/Desktop/gatton PV data/exogenous"
else:
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "/scratch/itee/uqsxu13/2020_data/2020_gatton_1"
    ABSOLUTE_IMG_DIR_2 = "/scratch/itee/uqsxu13/2020_data/2020_gatton_2"
    ABSOLUTE_INDEX_DIR = "/scratch/itee/uqsxu13/2020_data/2020_index"
    EXOGENOUS_DIR = "/scratch/itee/uqsxu13/2020_data/exogenous"

BATCH_SIZE = 5 if os.name == "nt" else 8
LEARNING_RATE = 0.0001 if os.name == "nt" else 0.00001
EPOCH = 10
VIT_MODEL_PATH = "VIT.pth"
MIX_MODEL_PATH = "mix.pth"
TRAIN_AGAIN = False
NUM_MIN = 5
TRAIN_STEP = 2
CURRENT_MODEL_PATH = VIT_MODEL_PATH if TRAIN_STEP == 1 else MIX_MODEL_PATH

DATASET_ARG = {
    "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "small": False,
    "output_length": 1,
    "mode": "",
    "input_length": NUM_MIN,
    "interval": 15,
    "tensor_size": (258, 258),
    "exogenous_folder": EXOGENOUS_DIR
}


def forward_model(device, data, optimizer, model, pre_model, criterion, is_train=False):
    inputs, labels, historical, exogenous = [element.to(device) for element in data]
    labels = labels.float()
    inputs = inputs.float()
    historical = historical.float()[:, :, None]
    exogenous = exogenous.float()

    if is_train:
        optimizer.zero_grad()

    if TRAIN_STEP == 1:
        outputs = model(inputs, exogenous)
    else:
        with torch.no_grad():
            pre_pred = pre_model(inputs, exogenous)
        outputs = model(historical, pre_pred)
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
    if TRAIN_STEP == 1:
        pre_model = None
        model = TimeVIT(num_min=NUM_MIN)
        if TRAIN_AGAIN:
            model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
    else:
        model = MixHistorical()
        pre_model = TimeVIT(num_min=NUM_MIN)
        pre_model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
        pre_model.eval()
        if TRAIN_AGAIN:
            model.load_state_dict(torch.load(MIX_MODEL_PATH, map_location=device))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)
    if pre_model is not None:
        pre_model.to(device)

    if (not os.path.exists(CURRENT_MODEL_PATH)) or (os.path.exists(CURRENT_MODEL_PATH) and TRAIN_AGAIN):

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
                                           model=model, pre_model=pre_model, criterion=criterion, is_train=True)
                running_train_loss += train_loss.item()
            avg_train_loss = running_train_loss / i

            model.eval()
            running_val_loss = 0.0
            for i, data in enumerate(validation_loader, 1):
                val_loss = forward_model(device=device, data=data, optimizer=optimizer,
                                         model=model, pre_model=pre_model, criterion=criterion, is_train=False)
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
                torch.save(model.state_dict(), "{}_best.pth".format(CURRENT_MODEL_PATH[:3]))
                best_loss = avg_val_loss

            records = records.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss},
                                     ignore_index=True)

        records.to_csv("training_records_{}epochs_vit_loss.csv".format(EPOCH), index=False)
        torch.save(model.state_dict(), CURRENT_MODEL_PATH)
        print("Finish!")
    else:
        print("There is existed model, start validation.")
        model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
        model.eval()

    print("Device is:", device)

    with torch.no_grad():
        pass
        # val_model_numeric(model, test_loader, device)


def val_model_numeric(model, data_loader, device):
    x = range(BATCH_SIZE * 5)
    ape_total = 0
    for i, data in enumerate(data_loader, 1):
        start = time.time()
        inputs, labels, dec_input, historical = [element.to(device) for element in data]
        inputs = inputs.float()
        historical = historical.float()
        dec_input = dec_input.float()
        dec_input = torch.cat((dec_input[:, :, None], torch.zeros(dec_input.size() + (1023,)).to(device)), 2)
        zero_dec_input = torch.zeros((inputs.size()[0], 5, 1024)).to(device)
        outputs = model(inputs, dec_input, historical)

        ape_total += ape_calculation(outputs.squeeze(), labels)

        if i == 1 and os.name == "nt":
            print("Spend time for 16 prediction: ", time.time() - start)
            outputs = rearrange(outputs, "b o -> (b o)")
            labels = rearrange(labels, "b o -> (b o)")
            plt.plot(x, outputs.numpy(), x, labels.numpy())
            plt.legend(["pred", "real data"])
            plt.title("Prediction of the first batch")
            plt.show()

        # print(outputs.squeeze())
        # print(labels)

    print("MAPE of the test set is:", (ape_total / len(data_loader.dataset)) * 100, "%")


def ape_calculation(pred, actual):
    return torch.sum(torch.abs((actual - pred) / actual)).item()


# def implement_second():
#


if __name__ == "__main__":
    torch.manual_seed(0)
    implement()

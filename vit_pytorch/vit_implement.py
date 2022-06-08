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
from nn_tool.raw_img_dataset import RawSkyImageDataset
from nn_tool.path_by_os import get_path

if os.name == "nt":
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
    ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
    ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"
else:
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "/scratch/itee/uqsxu13/2020_data/2020_gatton_1"
    ABSOLUTE_IMG_DIR_2 = "/scratch/itee/uqsxu13/2020_data/2020_gatton_2"
    ABSOLUTE_INDEX_DIR = "/scratch/itee/uqsxu13/2020_data/2020_index"

BATCH_SIZE = 4 if os.name == "nt" else 16
LEARNING_RATE = 0.0001 if os.name == "nt" else 0.00001
EPOCH = 25
SAVED_MODEL = "VIT.pth"
TRAIN_AGAIN = False
NUM_MIN = 5

DATASET_ARG = {
    "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "small": True,
    "output_length": 1,
    "mode": "",
    "input_length": NUM_MIN,
    "interval": 10,
}


def forward_model(device, data, dec_mask, optimizer, model, criterion, is_train=False):
    inputs, labels, historical = [element.to(device) for element in data]
    labels = labels.float()
    inputs = inputs.float()
    # dec_input = dec_input.float()
    historical = historical.float()[:, :, None]
    # dec_input_fill = repeat(dec_input, "b t -> b t h", h=dec_input.size()[1])
    # dec_input_fill = rearrange(dec_input_fill, "b t h -> b h t")
    # dec_input_fill = torch.masked_fill(dec_input_fill, dec_mask, value=0)
    # dec_input_fill = torch.cat((dec_input_fill,
    #                             torch.zeros(dec_input.size() + (1024 - dec_input.size()[1],)).to(device)), 2)

    # dec_input_fill = torch.cat((dec_input[:, :, None], torch.zeros(dec_input.size() + (1023,)).to(device)), 2)
    # zero_dec_input = torch.zeros((inputs.size()[0], 30, 1024)).to(device)

    if is_train:
        optimizer.zero_grad()

    outputs = model(inputs, historical)
    loss = criterion(outputs, labels)
    if is_train:
        loss.backward()
        optimizer.step()
    if os.name == "nt":
        print(loss.item())
        print(outputs)
        print(labels)
        print("------------" * 5)
    return loss


def implement():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimeVIT(num_min=NUM_MIN)
    if TRAIN_AGAIN:
        model.load_state_dict(torch.load(SAVED_MODEL, map_location=device))

    dataset = RawSkyImageDataset(**DATASET_ARG)

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
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    if (not os.path.exists(SAVED_MODEL)) or (os.path.exists(SAVED_MODEL) and TRAIN_AGAIN):

        records = pd.DataFrame(data={"epoch": [], "train_loss": [], "val_loss": []})
        print("Start training!")
        last_time = datetime.datetime.now()
        sys.stdout.flush()

        best_loss = 100

        for epoch in range(EPOCH):

            model.train()
            running_train_loss = 0.0
            dec_mask = ~torch.BoolTensor([[1, 0, 0, 0, 0],
                                          [1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0],
                                          [1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 1]]).to(device)
            for i, data in enumerate(train_loader, 1):
                train_loss = forward_model(device=device, data=data, dec_mask=dec_mask, optimizer=optimizer,
                                           model=model, criterion=criterion, is_train=True)
                running_train_loss += train_loss.item()
            avg_train_loss = running_train_loss / i

            model.eval()
            running_val_loss = 0.0
            for i, data in enumerate(validation_loader, 1):
                val_loss = forward_model(device=device, data=data, dec_mask=dec_mask, optimizer=optimizer,
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
                torch.save(model.state_dict(), "VIT_best.pth")
                best_loss = avg_val_loss

            records = records.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss},
                                     ignore_index=True)

        records.to_csv("training_records_{}epochs_vit_loss.csv".format(EPOCH), index=False)
        torch.save(model.state_dict(), SAVED_MODEL)
        print("Finish!")
    else:
        print("There is existed model, start validation.")
        model.load_state_dict(torch.load(SAVED_MODEL, map_location=device))
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


if __name__ == "__main__":
    torch.manual_seed(0)
    implement()

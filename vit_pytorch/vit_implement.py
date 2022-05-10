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
from nn_tool import mean_std_calculator
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

BATCH_SIZE = 4 if os.name == "nt" else 8
LEARNING_RATE = 0.001
EPOCH = 20

DATASET_ARG = {
    "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "small": True
}


def implement():
    # model = ViT(
    #     image_size=256,
    #     patch_size=32,
    #     num_classes=1,
    #     dim=1024,
    #     depth=4,
    #     heads=4,
    #     mlp_dim=2048,
    #     channels=18,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )

    model = TimeVIT()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists("VIT_original.pth"):

        records = pd.DataFrame(data={"epoch": [], "train_loss": [], "val_loss": []})
        print("Start training!")

        for epoch in range(EPOCH):

            model.train()
            running_train_loss = 0.0
            for i, data in enumerate(train_loader, 1):
                inputs, labels, dec_input, historical = [element.to(device) for element in data]
                labels = labels.float()
                inputs = inputs.float()
                dec_input = dec_input.float()
                dec_input = torch.cat((dec_input[:, :, None], torch.zeros(dec_input.size() + (1023, )).to(device)), 2)
                historical = historical.float()
                zero_dec_input = torch.zeros((inputs.size()[0], 5, 1024)).to(device)

                optimizer.zero_grad()

                outputs = model(inputs, dec_input, historical)
                train_loss = criterion(outputs, labels)
                running_train_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()
                if os.name == "nt":
                    print(train_loss.item())
            avg_train_loss = running_train_loss / i

            model.eval()
            running_val_loss = 0.0
            for i, data in enumerate(validation_loader, 1):
                inputs, labels, dec_input, historical = [element.to(device) for element in data]
                labels = labels.float()
                inputs = inputs.float()
                dec_input = dec_input.float()
                historical = historical.float()
                dec_input = torch.cat((dec_input[:, :, None], torch.zeros(dec_input.size() + (1023, )).to(device)), 2)
                zero_dec_input = torch.zeros((inputs.size()[0], 5, 1024)).to(device)
                outputs = model(inputs, dec_input, historical)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
            avg_val_loss = running_val_loss / i

            print("Epoch: {} / {}".format(epoch + 1, EPOCH))
            print("Average train loss: ", avg_train_loss)
            print("Average validation loss: ", avg_val_loss)
            sys.stdout.flush()
            if EPOCH - epoch < 10:
                torch.save(model.state_dict(), "TimeVIT_epoch{}_change_loss.pth".format(epoch + 1))

            records = records.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss},
                                     ignore_index=True)

        records.to_csv("training_records_{}epochs_vit_loss.csv".format(EPOCH), index=False)
        torch.save(model.state_dict(), "VIT_original_change_loss.pth")
        print("Finish!")
    else:
        print("There is existed model, start validation.")
        model.load_state_dict(torch.load("VIT_original.pth", map_location=device))
        model.eval()

    print("Device is:", device)

    with torch.no_grad():
        val_model_numeric(model, test_loader, device)


def val_model_numeric(model, data_loader, device):
    x = range(BATCH_SIZE * 5)
    ape_total = 0
    for i, data in enumerate(data_loader, 1):
        start = time.time()
        inputs, labels, dec_input, historical = [element.to(device) for element in data]
        inputs = inputs.float()
        historical = historical.float()
        dec_input = dec_input.float()
        dec_input = torch.cat((dec_input[:, :, None], torch.zeros(dec_input.size() + (1023, )).to(device)), 2)
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

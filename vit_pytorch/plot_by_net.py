import os
from os.path import join as path_join

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.utils.data

from vit import ViT, TimeVIT
from nn_tool.raw_img_dataset import RawSkyImageDataset

ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"

BATCH_SIZE = 1
OUTPUT_LENGTH = 1
DATASET_ARG = {
    "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_diff.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "interval": 5,
    "output_length": OUTPUT_LENGTH,
    "mode": "",
    "input_length": 4,
}


def plot_pred(start=0, end=1000, dec_out=False, output_length=OUTPUT_LENGTH):
    start = real_index(start, 1)
    end = real_index(end, 1)
    if os.path.exists("VIT.pth"):
        with torch.no_grad():
            # model = ViT(
            #     image_size=256,
            #     patch_size=32,
            #     num_classes=1,
            #     dim=1024,
            #     depth=6,
            #     heads=16,
            #     mlp_dim=2048,
            #     channels=18,
            #     dropout=0.1,
            #     emb_dropout=0.1
            # )
            model = TimeVIT(num_min=4)
            device = torch.device("cpu")
            dataset = RawSkyImageDataset(**DATASET_ARG)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            model.load_state_dict(torch.load("VIT.pth", map_location=device))
            model.eval()
            x = range(start * output_length, end * output_length)
            # x = range(start, end)
            output_array = numpy.array([])
            labels_array = numpy.array([])
            dec_array = numpy.array([])
            historical_array = numpy.array([])
            for i in range(start, end):
                data = data_loader.dataset.__getitem__(i)
                inputs, labels, historical = data
                inputs = inputs.float()
                historical = np.array(historical)
                # dec_input = np.array(dec_input)
                # dec_array = np.append(dec_array, dec_input)
                historical_array = np.append(historical_array, historical)
                historical = torch.from_numpy(historical).float()
                # dec_input = torch.from_numpy(dec_input).float()
                # dec_input = torch.cat((dec_input[:, None], torch.zeros(dec_input.size() + (1023,)).to(device)), 1)
                # zero_dec_input = torch.zeros((1, 30, 1024)).to(device)
                outputs = model(torch.unsqueeze(inputs, 0),
                                torch.unsqueeze(torch.unsqueeze(historical, 1), 0))
                print(outputs)
                output_array = np.concatenate((output_array, outputs[0].numpy()))
                labels_array = np.append(labels_array, labels)
                # if len(output_array) >= end:
                #     output_array = output_array[start: end]
                #     labels_array = labels_array[start: end]
                #     break

            if dec_out:
                historical_array = np.repeat(historical_array, 5)
                plt.plot(x, output_array, x, labels_array, x, historical_array)
                plt.legend(["pred", "real data", "historical"])
                plt.show()
                plt.plot(x, output_array, x, labels_array)
                plt.legend(["pred", "real data"])
                plt.show()
                plt.plot(x, output_array, x, historical_array)
                plt.legend(["pred",  "historical"])
                plt.show()
            else:
                plt.plot(x, output_array, x, labels_array)
                plt.legend(["pred", "real data"])
                # plt.title("90 minutes in 24-01-2020")
                plt.title("Prediction from {} to {}".format(start, end))
                plt.show()
    else:
        raise Exception("No saved model!")


def real_index(index, slip_step):
    return int(index / slip_step)


if __name__ == "__main__":
    plot_pred(start=4100, end=4900, dec_out=False, output_length=1)

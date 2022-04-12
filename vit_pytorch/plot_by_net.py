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

DATASET_ARG = {
    "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "interval": 2,
}


def plot_pred(start=0, end=1000):
    if os.path.exists("VIT_original.pth"):
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
            model = TimeVIT()
            device = torch.device("cpu")
            dataset = RawSkyImageDataset(**DATASET_ARG)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            model.load_state_dict(torch.load("VIT_original.pth", map_location=device))
            model.eval()
            x = range(start, end)
            output_array = numpy.array([])
            labels_array = numpy.array([])
            for i in range(start, end):
                data = data_loader.dataset.__getitem__(i)
                inputs, labels = data[0], data[1]
                inputs = inputs.float()
                outputs = model(torch.unsqueeze(inputs, 0))
                output_array = np.append(output_array, outputs.squeeze())
                labels_array = np.append(labels_array, labels)
                # if len(output_array) >= end:
                #     output_array = output_array[start: end]
                #     labels_array = labels_array[start: end]
                #     break

            plt.plot(x, output_array, x, labels_array)
            plt.legend(["pred", "real data"])
            plt.title("90 minutes in 24-01-2020")
            # plt.title("Prediction from {} to {}".format(start, end))
            plt.show()


if __name__ == "__main__":
    plot_pred(start=760, end=850)

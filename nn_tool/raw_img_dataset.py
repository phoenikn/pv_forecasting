import os
from os.path import join as path_join

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"


class RawSkyImageDataset(torch.utils.data.Dataset):
    """Use the raw Images"""

    def __init__(self, index_path, images_folder, interval: int = 1, transform=None, tensor_size=(256, 256)):
        """

        :param csv_path (string): Path of the csv for labels
        :param images_folder (string): Path of the image polder
        :param transform (transform): Optional transform
        """
        self.index_df = pd.read_csv(index_path)
        self.images_folder = images_folder
        self.tensor_size = tensor_size
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((1500, 1500)),
                transforms.Resize(tensor_size),
            ])
        self.transform = transform
        self.interval = interval
        self.datetime = self.index_df["DateTime"]
        self.labels = self.index_df["interval_{}".format(interval)]
        self.historical = self.index_df["Power(kW)"]

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, index):
        date_time = self.datetime[index]
        date, time = date_time.split("_")
        path_day = path_join(self.images_folder, date)
        img_stack = []
        for sec in range(0, 60, 10):
            img_name = date_time[:-2] + "{:02d}".format(sec) + ".jpg"
            img_path = path_join(path_day, img_name)
            img = Image.open(img_path)
            img = self.transform(img)

            img_stack.append(img)
            # img_stack = torch.cat((img_stack, img), 0)

        label = self.labels[index]
        img_stack = torch.stack(img_stack)
        historical = self.historical[index]

        return img_stack, label, historical


if __name__ == "__main__":
    test_ds = RawSkyImageDataset(path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"), ABSOLUTE_IMG_DIR_1)
    print(torch.tensor_split(test_ds.__getitem__(0)[0], 6, dim=0)[0].size())

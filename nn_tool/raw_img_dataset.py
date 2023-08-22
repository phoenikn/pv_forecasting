import os
from time import time as tm
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
EXOGENOUS_DIR = "C:/Users/s4544852/Desktop/gatton PV data/exogenous"


class RawSkyImageDataset(torch.utils.data.Dataset):
    """Use the raw Images"""

    def __init__(self, index_path, images_folder, exogenous_folder=EXOGENOUS_DIR, input_length: int = 5,
                 interval: int = 5,
                 hist_length=5, output_length: int = 5, transform=None, tensor_size=(256, 256), small=False,
                 mode="", index_folder=ABSOLUTE_INDEX_DIR):
        """

        :param csv_path (string): Path of the csv for labels
        :param images_folder (string): Path of the image polder
        :param transform (transform): Optional transform
        """
        rainfall_path = os.path.join(exogenous_folder, "gatton_2020_rainfall.csv")
        solar_irradiance_path = os.path.join(exogenous_folder, "gatton_2020_solar_irradiance.csv")
        temperature_path = os.path.join(exogenous_folder, "gatton_2020_temperature.csv")

        def df_norm(df, col):
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        self.index_df = pd.read_csv(index_path)
        self.rainfall_df = pd.read_csv(rainfall_path)
        self.irradiance_df = pd.read_csv(solar_irradiance_path)
        self.temperature_df = pd.read_csv(temperature_path)
        df_norm(self.rainfall_df, "Rainfall amount (millimetres)")
        df_norm(self.irradiance_df, "Daily global solar exposure (MJ/m*m)")
        df_norm(self.temperature_df, "Maximum temperature (Degree C)")

        if small:
            self.index_df = self.index_df[self.index_df.index < len(self.index_df) / 100]

        # if mode != "":
        #     weather_dir = path_join(index_folder, "date_{}.csv".format(mode))
        #
        #     if mode in ["sunny", "cloudy", "overcast"]:
        #         weather_idx = pd.read_csv(weather_dir)["old_idx"]
        #     else:
        #         raise Exception("Incorrect mode")
        #
        #     self.index_df = self.index_df[self.index_df.index.isin(weather_idx)]

        if mode == "sunny":
            self.index_df = self.index_df[self.index_df["DateTime"].str.startswith("2020-{:02d}-{:02d}".format(2, 20))]
        elif mode == "cloudy":
            self.index_df = self.index_df[self.index_df["DateTime"].str.startswith("2020-{:02d}-{:02d}".format(2, 21))]
        elif mode == "overcast":
            self.index_df = self.index_df[self.index_df["DateTime"].str.startswith("2020-{:02d}-{:02d}".format(3, 9))]
        elif mode == "rainy":
            self.index_df = self.index_df[self.index_df["DateTime"].str.startswith("2020-{:02d}-{:02d}".format(2, 4))]

        self.index_df = self.index_df.reset_index()

        self.images_folder = images_folder
        self.tensor_size = tensor_size
        if transform is None:
            transform = transforms.Compose([
                transforms.CenterCrop((1500, 1500)),
                transforms.Resize(tensor_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1119, 0.1167, 0.1461),
                                     (0.116, 0.1160, 0.1288))
            ])
        self.transform = transform
        self.input_length = input_length
        self.slip_window = output_length
        self.hist_length = hist_length
        self.interval = interval
        self.output_length = output_length
        self.datetime = self.index_df["DateTime"]
        self.labels = self.index_df["interval_{}".format(interval)]
        # self.dec_input = self.index_df["interval_{}".format(interval - 1)]
        self.historical = self.index_df["Power(kW)"]

    def __len__(self):
        return int(len(self.index_df) / self.slip_window) - self.input_length

    def __getitem__(self, index):
        index *= self.slip_window
        input_indexes = range(index, index + self.input_length)
        date_time = self.datetime[input_indexes]
        year, month, day = [int(x) for x in date_time.iloc[0].split("_")[0].split("-")]
        rainfall_row = self.rainfall_df[(self.rainfall_df["Month"] == month)
                                        & (self.rainfall_df["Day"] == day)]["Rainfall amount (millimetres)"]
        rainfall_value = rainfall_row.values[0]
        irradiance_row = self.irradiance_df[(self.rainfall_df["Month"] == month)
                                            & (self.rainfall_df["Day"] == day)]["Daily global solar exposure (MJ/m*m)"]
        irradiance_value = irradiance_row.values[0]
        temperature_row = self.temperature_df[(self.rainfall_df["Month"] == month)
                                              & (self.rainfall_df["Day"] == day)]["Maximum temperature (Degree C)"]
        temperature_value = temperature_row.values[0]
        img_stack = []
        start_time_min = 0
        for one_min in date_time:
            date, time = one_min.split("_")
            path_day = path_join(self.images_folder, date)
            if start_time_min == 0:
                ticks = [int(tick) for tick in time.split("-")]
                start_time_min = ticks[0] * 60 + ticks[1]
                start_time_min /= (60 * 24)
            for sec in range(0, 60, 10):
                img_name = one_min[:-2] + "{:02d}".format(sec) + ".jpg"
                img_path = path_join(path_day, img_name)
                img = Image.open(img_path)
                img = self.transform(img)
                img_stack.append(img)

        label = self.labels[range(index, index + self.output_length)]
        # dec_input = self.dec_input[indexes]
        img_stack = torch.stack(img_stack)
        historical = self.historical[range(index, index + self.hist_length)]
        exogenous = start_time_min, rainfall_value, irradiance_value, temperature_value
        exogenous = torch.tensor(exogenous)

        return img_stack, label.to_numpy(), historical.to_numpy(), exogenous


if __name__ == "__main__":
    DATASET_ARG = {
        "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
        "images_folder": ABSOLUTE_IMG_DIR_1,
        "small": False,
        "output_length": 1,
        "mode": "cloudy",
        "input_length": 5,
        "interval": 10,
    }
    test_ds = RawSkyImageDataset(**DATASET_ARG)
    print(len(test_ds.__getitem__(0)))

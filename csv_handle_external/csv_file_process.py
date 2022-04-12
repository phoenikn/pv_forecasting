import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]
PV_FILE_NAME = "AQG1_B001_PM001.Sts.P_kW.csv"


def combine_csv_files():
    data_list = []
    for month in MONTHS:
        folder = os.path.join(ABSOLUTE_FILE_DIR, month)
        file = os.path.join(folder, PV_FILE_NAME)
        if os.path.exists(file):
            data = pd.read_csv(file)
            data_list.append(data)

    path_save = os.path.join(ABSOLUTE_FILE_DIR, "pv_2020.csv")

    all_data = pd.concat(data_list, axis=0, ignore_index=True)
    all_data.rename(columns={"AQG1_B001_PM001.Sts.P_kW": "Power(kW)"}, inplace=True)
    all_data.to_csv(path_save, index=False)


def align_with_interval(output_name="data_2020_interval", file_name="data_2020.csv",
                        file_dir=ABSOLUTE_FILE_DIR, index_dir=ABSOLUTE_INDEX_DIR):
    data = pd.read_csv(os.path.join(file_dir, file_name))
    output_col = data["Power(kW)"]
    for i in range(-1, -6, -1):
        shifted = output_col.shift(periods=i)
        data["interval_{}".format(-i)] = shifted
    data.dropna(inplace=True)
    data.to_csv(os.path.join(index_dir, output_name + ".csv"), index=False)
    data.to_csv(os.path.join(file_dir, output_name + ".csv"), index=False)


if __name__ == "__main__":
    # combine_csv_files()
    # data_all = pd.read_csv(os.path.join(ABSOLUTE_FILE_DIR, "data_2020_interval.csv"))
    # plt.plot(data_all["Power(kW)"][0:10000])
    # plt.show()
    # align_with_interval()
    pass

import os

import torch
import torch.utils.data

from nn_tool.raw_img_dataset import RawSkyImageDataset

ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"
DATASET_ARG = {
    "index_path": os.path.join(ABSOLUTE_INDEX_DIR, "data_2020_diff.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "small": True,
    "pred_horizon": 5,
    "mode": "",
}


def calculate(dataset: torch.utils.data.Dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset) // 10, shuffle=True)

    data = next(iter(loader))
    mean = torch.mean(data[0], dim=[0, 1, 3, 4])
    std = torch.std(data[0], dim=[0, 1, 3, 4])

    print("Mean of the dataset is:", mean.tolist())
    print("Std of the dataset is:", std.tolist())

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    calculate(RawSkyImageDataset(**DATASET_ARG))


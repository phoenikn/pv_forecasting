import os

from custom_dataset import PvImageFeatureDataset
from custom_dataset_3channel import PvImage3ChannelFeatureDataset
from custom_dataset_4channel import PvImage4ChannelFeatureDataset
from custom_dataset_3image_8stack import PvImage8ChannelFeatureDataset
from custom_dataset_two_grayscale import PvImage2ChannelFeatureDataset

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import math
from custom_res50 import resnet50
from custom_resnet101 import resnet101


def calculate():
    index_dir = "../index/combined_ordinary_cliff_10min_three.csv"
    ordinary = "../extracted_10min_three"
    cliff = "../extracted_cliff_three"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    dataset = PvImage2ChannelFeatureDataset(index_dir, ordinary, cliff, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    data = next(iter(loader))
    mean = torch.mean(data[0], dim=[0, 2, 3])
    std = torch.std(data[0], dim=[0, 2, 3])

    print(mean.tolist())
    print(std.tolist())


# def get_mean_std(loader):
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#     for data, _ in loader:
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1
#     mean = channels_sum / num_batches
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
#     return mean, std


if __name__ == "__main__":
    calculate()
    # index_dir = "../index/combined_ordinary_cliff.csv"
    # ordinary = "../extracted_1min_before_cliff"
    # cliff = "../extracted_cliff"
    #
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224))
    # ])
    #
    # dataset = PvImageFeatureDataset(index_dir, ordinary, cliff, transform=transform)
    #
    # loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    # mean, std = get_mean_std(loader)
    # print(mean)
    # print(std)


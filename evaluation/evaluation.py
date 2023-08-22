import os
import sys

import torch
import torch.utils.data
import torchmetrics as tm
import torch.nn as nn
import torchvision
from torchvision import transforms
from os.path import join as path_join

from einops import rearrange

from nn_tool.raw_img_dataset import RawSkyImageDataset
from convLSTM.convlstm import ConvLSTMOut

if os.name == "nt":
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
    ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
    ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"
    EXOGENOUS_DIR = "C:/Users/s4544852/Desktop/gatton PV data/exogenous"
else:
    ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
    ABSOLUTE_IMG_DIR_1 = "/scratch/itee/uqsxu13/2020_data/pre_compress"
    ABSOLUTE_IMG_DIR_2 = "/scratch/itee/uqsxu13/2020_data/2020_gatton_2"
    ABSOLUTE_INDEX_DIR = "/scratch/itee/uqsxu13/2020_data/2020_index"
    EXOGENOUS_DIR = "/scratch/itee/uqsxu13/2020_data/exogenous"

INTERVAL = 5

BATCH_SIZE = 8
EPOCH = 5
NUM_MIN = 5
MODEL_NAMES = ["ResNet", "ConvLSTM"]
METRIC_NAMES = ["MAE", "MAPE", "RMSE", "R2 Value"]
MODES = ["", "sunny", "cloudy", "overcast", "rainy"]


def eval_implement(interval, mode):
    DATASET_ARG_224 = {
        "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
        "images_folder": ABSOLUTE_IMG_DIR_1,
        "output_length": 1,
        "mode": mode,
        "input_length": NUM_MIN,
        "interval": interval + 5,
        # the real interval + 5
        "tensor_size": (256, 256),
        "exogenous_folder": EXOGENOUS_DIR,
        "transform": transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1119, 0.1167, 0.1461),
                                 (0.116, 0.1160, 0.1288))
        ])
    }

    DATASET_ARG_256 = {
        "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
        "images_folder": ABSOLUTE_IMG_DIR_1,
        "output_length": 1,
        "mode": mode,
        "input_length": NUM_MIN,
        "interval": interval + 5,
        # the real interval + 5
        "tensor_size": (256, 256),
        "exogenous_folder": EXOGENOUS_DIR,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1119, 0.1167, 0.1461),
                                 (0.116, 0.1160, 0.1288))
        ])
    }

    torch.manual_seed(0)

    metrics = [[tm.MeanAbsoluteError(),
                tm.MeanAbsolutePercentageError(),
                tm.MeanSquaredError(squared=False),
                tm.R2Score()] for i in range(2)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_metrics in metrics:
        for metric in model_metrics:
            metric.to(device)


    def get_loader(arg):
        dataset = RawSkyImageDataset(**arg)
        if mode == "":
            all_data_size = len(dataset)
            train_size = int(0.6 * all_data_size)
            validation_size = int(0.2 * all_data_size)
            test_size = all_data_size - train_size - validation_size
            _, _, test_set = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
            return torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    test_loader_224 = get_loader(DATASET_ARG_224)
    test_loader_256 = get_loader(DATASET_ARG_256)


    with torch.no_grad():
        resnet = torchvision.models.resnet50(pretrained=False)
        fc_features = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(90, 64, kernel_size=(7, 7))
        resnet.fc = nn.Linear(fc_features, 1)
        resnet.load_state_dict(torch.load("RESNET_best_interval{}.pth".format(interval), map_location=device))
        resnet.to(device)
        resnet.eval()

        convlstm = ConvLSTMOut(5, {
            "input_dim": 3,
            "hidden_dim": [4, 4],
            "kernel_size": (3, 3),
            "num_layers": 2,
            "batch_first": True,
            "bias": True,
            "return_all_layers": False
        })
        convlstm.load_state_dict(torch.load("CONVLSTM_best_interval{}.pth".format(interval), map_location=device))
        convlstm.to(device)
        convlstm.eval()

        for i, data in enumerate(test_loader_224, 1):
            inputs, labels, historical, exogenous = [element.to(device) for element in data]
            labels = labels.float()
            inputs = inputs.float()
            inputs = rearrange(inputs, "b i c h w -> b (i c) h w")
            # historical = historical.float()[:, :, None]
            # exogenous = exogenous.float()
            res_pred = resnet(inputs)

            for metric in metrics[0]:
                metric(res_pred, labels)

            if i % 100 == 0:
                print(i / len(test_loader_224) * 100, "%")
                sys.stdout.flush()

        print("Start ConvLSTM part")
        print("--------------------------------")

        for i, data in enumerate(test_loader_256, 1):
            inputs, labels, historical, exogenous = [element.to(device) for element in data]
            labels = labels.float()
            inputs = inputs.float()
            # historical = historical.float()[:, :, None]
            # exogenous = exogenous.float()
            cl_pred = convlstm(inputs)

            for metric in metrics[1]:
                metric(cl_pred, labels)
            if i % 100 == 0:
                print(i / len(test_loader_256) * 100, "%")
                sys.stdout.flush()

        print("Interval:", interval, " / Mode:", "all" if mode == "" else mode)

        for model_index, model_name in enumerate(MODEL_NAMES):
            print(model_name, ":")
            for metric_index, metric_name in enumerate(METRIC_NAMES):
                print(metric_name, ":", metrics[model_index][metric_index].compute().item())


if __name__ == "__main__":
    # for weather_mode in MODES:
    #     eval_implement(INTERVAL, weather_mode)
    eval_implement(5, "rainy")

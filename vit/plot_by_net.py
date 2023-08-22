import os
import time
from os.path import join as path_join

import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import numpy
import numpy as np
import torch
import torch.utils.data

from einops import rearrange

from vit import ViT, TimeVIT
from nn_tool.raw_img_dataset import RawSkyImageDataset
from historical_mix_model import MixHistorical
from baseline_models.scnn import SCNN

ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"

BATCH_SIZE = 1
OUTPUT_LENGTH = 1
DATASET_ARG = {
    "index_path": path_join(ABSOLUTE_INDEX_DIR, "data_2020_interval.csv"),
    "images_folder": ABSOLUTE_IMG_DIR_1,
    "interval": 10,
    "output_length": OUTPUT_LENGTH,
    "mode": "",
    "input_length": 5,
    "tensor_size": (258, 258),
}


def get_output(data, pre_model, model):
    inputs, labels, historical, exogenous = data
    inputs = inputs.float()
    exogenous = exogenous.float()
    historical = np.array(historical)
    historical = torch.from_numpy(historical).float()
    outputs_pre = pre_model(torch.unsqueeze(inputs, 0), exogenous.unsqueeze(0))
    outputs = model(torch.unsqueeze(torch.unsqueeze(historical, 1), 0), outputs_pre)

    return outputs


def plot_pred(month, day, hour, minute, start=0, end=1000):
    start = real_index(start, 1) - 12
    end = real_index(end, 1) - 12
    if os.path.exists("VIT.pth"):
        with torch.no_grad():

            device = torch.device("cpu")
            dataset = RawSkyImageDataset(**DATASET_ARG)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            model = MixHistorical()
            pre_model = TimeVIT(num_min=5)
            model_10 = MixHistorical()
            pre_model_10 = TimeVIT(num_min=5)
            model_15 = MixHistorical()
            pre_model_15 = TimeVIT(num_min=5)

            pre_model.load_state_dict(torch.load("VIT.pth", map_location=device))
            model.load_state_dict(torch.load("mix.pth", map_location=device))

            pre_model_10.load_state_dict(torch.load("10 interval records/models/VIT.pth", map_location=device))
            model_10.load_state_dict(torch.load("10 interval records/models/mix.pth", map_location=device))

            pre_model_15.load_state_dict(torch.load("15 interval records/models/VIT.pth", map_location=device))
            model_15.load_state_dict(torch.load("15 interval records/models/mix.pth", map_location=device))

            model.eval()
            pre_model.eval()
            model_10.eval()
            pre_model_10.eval()
            model_15.eval()
            pre_model_15.eval()

            # x = range(0, end * output_length - start * output_length)
            output_array = numpy.array([])
            output_10_array = numpy.array([])
            output_15_array = numpy.array([])
            labels_array = numpy.array([])
            flag = True
            for i in range(start, end):
                if flag:
                    start_time = time.time()
                data = data_loader.dataset.__getitem__(i)
                data_10 = data_loader.dataset.__getitem__(i-5)
                data_15 = data_loader.dataset.__getitem__(i-15)

                # print(outputs)

                outputs = get_output(data, pre_model, model)
                # outputs_10 = get_output(data_10, pre_model_10, model_10)
                # outputs_15 = get_output(data_15, pre_model_15, model_15)

                _, labels, historical, _ = data

                output_array = np.concatenate((output_array, outputs[0].numpy()))
                # output_10_array = np.concatenate((output_10_array, outputs_10[0].numpy()))
                # output_15_array = np.concatenate((output_15_array, outputs_15[0].numpy()))
                labels_array = np.append(labels_array, labels)
                if flag:
                    print("Time usage for 1 pred: ", time.time() - start_time)
                    flag = False
                print(((i - start) / (end - start)) * 100, "%")

            fig = plt.figure(figsize=(20, 7.5))
            plt.title("PV generation forecasting for {:02d}-{:02d}-2020".format(day, month), fontsize=20)
            plt.xlabel("Time", fontsize=20)
            plt.ylabel("Normalized PV Generation", fontsize=20)
            fig.gca().xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
            # fig.gca().xaxis.set_major_locator(md.MinuteLocator(byminute=range(0, 60, 10)))
            fig.gca().xaxis.set_major_locator(md.MinuteLocator(byminute=[0, 30]))
            time_axis = [datetime.datetime(year=2020, month=month, day=day,
                                           hour=hour, minute=minute)
                         + datetime.timedelta(minutes=j) for j in range(end - start)]

            plt.plot(time_axis, output_array, linewidth=1.5, linestyle="--")
            # plt.plot(time_axis, output_10_array, linewidth=1.5, linestyle="-.")
            # plt.plot(time_axis, output_15_array, linewidth=1.5, linestyle=":")
            plt.plot(time_axis, labels_array, linewidth=1.5)
            plt.legend(["Our Model 5 min", "Real Data"], prop={'size': 15})
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            # plt.title("Prediction from time step {} to {}".format(start, end))
            plt.grid()
            plt.show()
            # plt.savefig("FIG_COMPARE_13-15.eps", dpi=600, format="eps")
    else:
        raise Exception("No saved model!")


def real_index(index, slip_step):
    return int(index / slip_step)


if __name__ == "__main__":
    torch.manual_seed(0)
    plot_pred(1, 28, 13, 30, start=3825, end=3975)  # 327 - 1050
    # plot_pred(4, 9, 6, 37, start=52057, end=52694)  # 327 - 1050
    # plot_pred(start=20623, end=20713)  # 327 - 1050
    # plot_pred(start=8410, end=8510)  # 327 - 1050

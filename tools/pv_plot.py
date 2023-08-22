import os
import time
from os.path import join as path_join

import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import numpy as np
import pandas as pd


def pv_plot(month, day):
    pv_csv = pd.read_csv("C:/Users/s4544852/Desktop/gatton PV data/index_2020/data_2020_interval.csv")
    fig = plt.figure(figsize=(10, 5))
    plt.title("PV generation: {:02d}-{:02d}-2020".format(day, month))
    # plt.xlabel("Time (Year)")
    # plt.ylabel("PV installations size (MW)")
    pv_selected = pv_csv[pv_csv["DateTime"].str.startswith("2020-{:02d}-{:02d}".format(month, day))]
    power_selected = pv_selected["Power(kW)"]
    starting_hour = int(pv_selected["DateTime"].iloc[0][11:13])
    starting_min = int(pv_selected["DateTime"].iloc[0][14:16])
    fig.gca().xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
    fig.gca().xaxis.set_major_locator(md.HourLocator())
    time_axis = [datetime.datetime(year=2007, month=month, day=day)
                 + datetime.timedelta(minutes=j + starting_hour * 60 + starting_min) for j in range(len(pv_selected))]
    # time_axis = [datetime.datetime(year=2020, month=4, day=1)
    #              + datetime.timedelta(days=int(j/665)) for j in range(50072 - 46747)]
    plt.plot(time_axis, power_selected)
    plt.legend(["PV Generation"])
    plt.show()


# for i in range(20, 21):
#     pv_plot(2, i)

pv_plot(2, 4)
# for i in range(23, 30):
#     pv_plot(1, i)
# pv_plot(3, 9)

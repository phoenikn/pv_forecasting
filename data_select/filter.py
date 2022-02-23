import pre_processing.pv_data_process as pp
import numpy as np
import pandas as pd


def data_filter(pv_data_dir: str, time_interval: int = 30, threshold: int = -0.1):
    pv_data = pp.NormPvData(pv_data_dir)
    data = pv_data.get_powers()
    data = data[:-time_interval]
    interval = data[::time_interval]
    difference = interval.diff()
    difference_filtered = difference[difference < threshold]
    time = pv_data.get_datetime()
    time_first = time[difference_filtered.index - time_interval - 10].reset_index(drop=True)
    time_second = time[difference_filtered.index - time_interval].reset_index(drop=True)
    time_pair = pd.concat([time_first, time_second], axis=1)
    time_pair.columns = ["first", "second"]
    return time_pair


def near_cliff_data_select(pv_data_dir: str, time_interval: int = 30, threshold: int = -0.1):
    pv_data = pp.NormPvData(pv_data_dir)
    data = pv_data.get_powers()
    data = data[:-time_interval]
    interval = data[::time_interval]
    difference = interval.diff()
    difference_filtered = difference[difference < threshold]
    time = pv_data.get_datetime()
    # print(difference_filtered)
    time_first = time[difference_filtered.index - time_interval - 10 - 600].reset_index(drop=True)
    time_second = time[difference_filtered.index - time_interval - 600].reset_index(drop=True)
    time_pair = pd.concat([time_first, time_second], axis=1)
    time_pair.columns = ["first", "second"]
    return time_pair


def combine_index(cliff_csv_dir, ordinary_csv_dir):
    _cliff = pd.read_csv(cliff_csv_dir)
    _ordinary = pd.read_csv(ordinary_csv_dir)
    _cliff["label"] = pd.Series(([1] * _cliff.shape[0]))
    _ordinary["label"] = pd.Series(([0] * _ordinary.shape[0]))
    combined = pd.concat([_cliff, _ordinary], axis=0).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    ordinary = "../index/10min_before_cliff.csv"
    cliff = "../index/cliff_time_points.csv"
    all_three_months = "../pv_data/data_has_img_and_value.csv"
    # all_time = data_filter(all_three_months)
    # all_time.to_csv("../index/cliff_time_points.csv", index=False)
    # all_time_1min_before = near_cliff_data_select(all_three_months)
    # all_time_1min_before.to_csv("../index/1min_before_cliff.csv", index=False)

    # all_time_10min_before = near_cliff_data_select(all_three_months)
    # all_time_10min_before.to_csv("../index/10min_before_cliff.csv", index=False)
    combine_index(cliff, ordinary).to_csv("../index/combined_ordinary_cliff_10min.csv", index=False)

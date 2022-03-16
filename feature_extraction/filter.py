import pre_processing.pv_data_process as pp
import numpy as np
import pandas as pd


def find_time_points(pv_data_dir: str, time_interval: int = 30, threshold: int = -0.1):
    pv_data = pp.NormPvData(pv_data_dir)
    data = pv_data.get_powers()
    data = data[:-time_interval]
    interval = data[::time_interval]
    difference = interval.diff()
    difference_filtered = difference[difference < threshold]
    time = pv_data.get_datetime()
    return time, difference_filtered


def data_filter(pv_data_dir: str, time_interval: int = 30, threshold: int = -0.1):
    time, difference_filtered = find_time_points(pv_data_dir, time_interval, threshold)
    time_first = time[difference_filtered.index - time_interval - 10].reset_index(drop=True)
    time_second = time[difference_filtered.index - time_interval].reset_index(drop=True)
    time_pair = pd.concat([time_first, time_second], axis=1)
    time_pair.columns = ["first", "second"]
    return time_pair


def data_filter_n(pv_data_dir, time_interval: int = 30, sample_interval: int = 10, threshold: int = -0.1,
                  frame_amount: int = 3, numeric: bool = True, before_interval: int = 0):
    time, difference_filtered = find_time_points(pv_data_dir, time_interval, threshold)

    columns = []
    for i in reversed(range(frame_amount)):
        time_column = time[difference_filtered.index
                           - time_interval - sample_interval * i - before_interval].reset_index(drop=True)
        columns.append(time_column)

    time_selected = pd.concat(columns, axis=1)
    time_selected.columns = range(1, frame_amount + 1)
    if numeric:
        time_selected["value"] = difference_filtered.reset_index(drop=True)
    return time_selected


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
    ordinary_three = "../index/10min_before_cliff_three.csv"
    cliff_three = "../index/cliff_time_points_three.csv"

    all_three_months = "../pv_data/data_has_img_and_value.csv"

    all_time = data_filter_n(all_three_months, frame_amount=5, sample_interval=20)
    all_time.to_csv("../index/test.csv", index=False)
    # all_time_1min_before = near_cliff_data_select(all_three_months)
    # all_time_1min_before.to_csv("../index/1min_before_cliff.csv", index=False)

    # all_time_10min_before = near_cliff_data_select_three(all_three_months)
    # all_time_10min_before.to_csv("../index/10min_before_cliff_three.csv", index=False)
    # combine_index(cliff_three, ordinary_three).to_csv("../index/combined_ordinary_cliff_10min_three.csv", index=False)

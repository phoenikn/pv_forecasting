import os
import numpy as np
import pandas as pd
from filter import *
from image_extract import *

PV_DATA_DIR = "../pv_data/data_has_img_and_value.csv"

FILTER_ARGS = {
    "pv_data_dir": PV_DATA_DIR,
    "time_interval": 60,
    "sample_interval": 10,
    "threshold": -0.1,
    "frame_amount": 6,
    "numeric": True,
    "before_interval": 0,
}

INDEX_FOLDER = "../index"
INDEX_NAME = "decreased_data"
DATA_FOLDER = "../extracted_data/decreased"


def generate_data(filter_args: dict, index_folder: str, index_name: str, data_folder: str):
    filtered_datetime = data_filter_n(**filter_args)
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    index_path = os.path.join(index_folder, index_name + ".csv")
    filtered_datetime.to_csv(index_path, index=False)

    extract_n_gray_arr(index_path, data_folder)

    print("Select data and extract feature successfully!")


if __name__ == "__main__":
    generate_data(FILTER_ARGS, INDEX_FOLDER, INDEX_NAME, DATA_FOLDER)

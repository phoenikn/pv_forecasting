import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tools.util import get_path_from_datetime
from motion_detection import piv


def extract_one_image_pair(index_one, index_two):
    path1 = get_path_from_datetime(index_one)
    path2 = get_path_from_datetime(index_two)
    detector = piv.PivDetector(path1, path2)
    velocity1, velocity2 = detector.piv()
    velocity1[np.isnan(velocity1)] = 0
    velocity2[np.isnan(velocity2)] = 0
    cloud = detector.get_cloud_bi()
    sun_matrix = detector.get_sun_matrix()

    # print(sun_matrix.shape, cloud.shape, velocity1.shape, velocity2.shape)

    return sun_matrix, cloud, velocity1, velocity2


def extract_from_csv(csv_dir: str):
    start = time.time()
    csv_index = pd.read_csv(csv_dir)
    for index, row in csv_index.iterrows():
        sun_matrix, cloud, velocity1, velocity2 = extract_one_image_pair(row["first"], row["second"])
        np.savez("../extracted_10min_before_cliff/" + row["second"] + ".npz", sun_matrix=sun_matrix, cloud=cloud,
                 velocity1=velocity1, velocity2=velocity2)
        print((index + 1) * 100 / csv_index.shape[0], "%")
    print(time.time()-start)


if __name__ == "__main__":
    # motion_test_1 = '2018-07-04_12-08-40'
    # motion_test_2 = '2018-07-04_12-08-30'
    # extract_from_csv("../index/1min_before_cliff.csv")
    extract_from_csv("../index/10min_before_cliff.csv")
    # a = np.load("../extracted_cliff/2018-07-01_08-15-00.npz")
    # print(a["cloud"])

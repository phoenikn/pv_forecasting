import os
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tools.util import get_path_from_datetime
from cloud_detection.cloud_detection import SkyImage
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
    gray_img = detector.get_image_gray()

    # print(sun_matrix.shape, cloud.shape, velocity1.shape, velocity2.shape)

    return sun_matrix, cloud, velocity1, velocity2, gray_img

def extract_gray_arr(datetime):
    path = get_path_from_datetime(datetime)
    image = SkyImage(path)
    return np.asarray(image.get_image_gray())


def extract_three_image(index_one, index_two, index_three):
    first_pair = extract_one_image_pair(index_one, index_two)
    second_pair = extract_one_image_pair(index_two, index_three)
    return first_pair, second_pair


def extract_from_csv(csv_dir: str):
    start = time.time()
    csv_index = pd.read_csv(csv_dir)
    for index, row in csv_index.iterrows():
        sun_matrix, cloud, velocity1, velocity2, gray_img = extract_one_image_pair(row["first"], row["second"])
        np.savez("../extracted_10min_before_cliff_with_gray/" + row["second"] + ".npz", sun_matrix=sun_matrix,
                 cloud=cloud,
                 velocity1=velocity1, velocity2=velocity2, gray_img=gray_img)
        print((index + 1) * 100 / csv_index.shape[0], "%")
    print(time.time() - start)


def extract_from_csv_three(csv_dir: str):
    """Extract 8 features from adjacent three images"""
    start = time.time()
    csv_index = pd.read_csv(csv_dir)
    for index, row in csv_index.iterrows():
        feature_pair_one, feature_pair_two = extract_three_image(row["first"], row["second"], row["third"])
        sun_matrix1, cloud1, velocity1, velocity2, gray_img1 = feature_pair_one
        sun_matrix2, cloud2, velocity3, velocity4, gray_img2 = feature_pair_two
        np.savez("../extracted_cliff_three/" + row["third"] + ".npz",
                 sun_matrix1=sun_matrix1, cloud1=cloud1,
                 velocity1=velocity1, velocity2=velocity2, gray_img1=gray_img1,
                 sun_matrix2=sun_matrix2, cloud2=cloud2,
                 velocity3=velocity3, velocity4=velocity4, gray_img2=gray_img2)
        print((index + 1) * 100 / csv_index.shape[0], "%")
    print(time.time() - start)


def extract_n_gray_arr(csv_dir: str, data_folder: str):
    csv_index = pd.read_csv(csv_dir)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    for index, row in csv_index.iterrows():
        gray_img_dict = {"gray_img{}".format(i): extract_gray_arr(datetime) for i, datetime in enumerate(row[:-1], 1)}
        data_path = os.path.join(data_folder, row[-2] + ".npz")
        np.savez(data_path, **gray_img_dict)
        print((index + 1) * 100 / csv_index.shape[0], "%")


if __name__ == "__main__":
    motion_test_1 = '2018-07-04_12-08-40'
    motion_test_2 = '2018-07-04_12-08-30'
    # print(extract_one_image_pair(motion_test_1, motion_test_2))
    # extract_n_gray_arr("../index/cliff_time_points_three_numeric.csv")
    # extract_from_csv("../index/10min_before_cliff.csv")
    # a = np.load("../extracted_data/test/2018-07-01_08-15-00.npz")
    # plt.imshow(a["gray_img1"], cmap="gray")
    # plt.show()

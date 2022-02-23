import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import math
import time

import sun_positioning.sun_positioning as sp

# CROP_SIZE = (271, 0, 1822, 1516)
RESIZE = 512
IMG_SIZE = (RESIZE, RESIZE)
# INSCRIBED_SQUARE = (75, 75, 437, 437)
INSCRIBED_SQUARE = (74, 74, 438, 438)
# CROP_SIDE = 362
CROP_SIDE = 364
SEC_CROP = (CROP_SIDE, CROP_SIDE)

BRBG_THRESHOLD = 2.5
BRBG_THRESHOLD_MINMAX = 0.5


class SkyImage:

    def __init__(self, image_dir: str, mode: str = "RGB", resize=IMG_SIZE):
        # self._image = Image.open(image_dir).convert(mode)
        self._drawer = sp.SunPositionDraw(image_dir)
        self._image = self._drawer.draw_circle()

        self._image_original = image_dir
        self._mode = mode
        # self._image = self._image.crop(CROP_SIZE)
        self._image = self.img_resize(resize)

        # Calculate the sun center after resize
        self.sun_center = self._drawer.get_sun()
        self.sun_center = ((self.sun_center[0] / 1551) * 512, (self.sun_center[1] / 1516) * 512)

        self._image = self.handle_zero_value()
        # self._image = self.fix_low_value()
        self._image = self.crop_to_square()

        # Calculate the sun center after crop
        self.sun_center = (round(self.sun_center[0] - 74), round(self.sun_center[1] - 74))

    def get_image(self):
        return self._image

    def handle_zero_value(self, value: int = 1):
        plus_value = np.vectorize(lambda x: x + value if x == 0 else x)
        return plus_value(self._image)

    def fix_low_value(self, value: int = 30):
        plus_value = np.vectorize(lambda x: 255 if x + value >= 255 else x + value)
        return plus_value(self._image)

    def show_original(self):
        image_original = Image.open(self._image_original).convert(self._mode)
        plt.imshow(image_original)
        plt.show()

    def show_brighter(self):
        plt.imshow(self.fix_low_value())
        plt.show()

    def show_resized(self):
        plt.imshow(self._image)
        plt.show()

    def show_gray(self):
        plt.imshow(self._image, cmap="gray")
        plt.show()

    def split_to_rgb(self, image: Image = None):
        if image is None:
            image = self._image
        red, green, blue = np.dsplit(np.asarray(image), 3)
        red = np.asarray(red).reshape(SEC_CROP)
        green = np.asarray(green).reshape(SEC_CROP)
        blue = np.asarray(blue).reshape(SEC_CROP)
        return [red, green, blue]

    @staticmethod
    def min_max(target):
        target_min, target_max = target.min(), target.max()
        return (target - target_min) / (target_max - target_min)

    def brbg_ratio(self):
        red, green, blue = self.split_to_rgb()
        br_ratio = blue / red
        bg_ratio = blue / green
        return br_ratio + bg_ratio

    def minmax_brbg_ratio(self):
        return self.min_max(self.brbg_ratio())

    def show_minmax_brbg(self):
        brbg = self.minmax_brbg_ratio()
        plt.imshow(brbg * 255, cmap="gray_r", vmax=255, vmin=0)
        plt.title("brbg")
        plt.show()

    def save_brbg(self, name):
        brbg = self.minmax_brbg_ratio()
        brbg *= 255
        plt.imsave(name, brbg, cmap="Greys")

    def threshold_on_brbg(self):
        brbg = self.brbg_ratio()
        threshold = np.vectorize(lambda x: 1 if x >= BRBG_THRESHOLD else 0)
        return threshold(brbg)

    def threshold_on_brbg_minmax(self):
        brbg = self.minmax_brbg_ratio()
        threshold = np.vectorize(lambda x: 1 if x >= BRBG_THRESHOLD_MINMAX else 0)
        return threshold(brbg)

    def show_threshold(self):
        threshold_brbg = self.threshold_on_brbg() * 255
        plt.imshow(threshold_brbg, cmap="gray_r", vmax=255, vmin=0)
        plt.title("threshold on brbg")
        plt.show()

    def show_threshold_minmax(self):
        threshold_brbg = self.threshold_on_brbg_minmax() * 255
        plt.imshow(threshold_brbg, cmap="gray_r", vmax=255, vmin=0)
        plt.title("threshold on brbg with minmax")
        plt.show()

    def save_threshold_brbg(self, name):
        plt.imsave(name, self.threshold_on_brbg() * 255, cmap="Greys")

    def blue_minus_red(self):
        red, green, blue = self.split_to_rgb()
        return blue - red

    def minmax_b_minus_r(self):
        return self.min_max(self.blue_minus_red())

    def show_minmax_b_minus_r(self):
        b_m_r = self.minmax_b_minus_r()
        plt.imshow(b_m_r * 255, cmap="gray_r", vmax=255, vmin=0)
        plt.title("bmr")
        plt.show()

    def img_resize(self, size):
        return self._image.resize(size)

    def kmeans(self, processed_img=None, image_size=IMG_SIZE):
        if processed_img is None:
            processed_img = self.minmax_brbg_ratio()
        reshape = processed_img.reshape(processed_img.shape[0] * processed_img.shape[1], 1)
        kmeans = KMeans(n_clusters=3).fit(reshape)
        clusters = kmeans.labels_.reshape(image_size)
        clusters_show = clusters * (255 / 3)
        im_cluster = Image.fromarray(clusters_show)
        plt.imshow(im_cluster)
        plt.show()

    def crop_to_square(self):
        img = Image.fromarray(self._image.astype(np.uint8))
        img = img.crop(INSCRIBED_SQUARE)
        return img

    def get_sun_center(self):
        return self.sun_center


if __name__ == "__main__":
    heavy_cloud = "../sky_image/2018-07-02/2018-07-02_09-43-20.jpg"
    sink_cloud = "../sky_image/2018-07-11/2018-07-11_08-37-00.jpg"
    motion_test_1 = '../sky_image/2018-07-04/2018-07-04_12-08-40.jpg'
    motion_test_2 = '../sky_image/2018-07-04/2018-07-04_12-08-30.jpg'
    motion_test_3 = '../sky_image/2018-07-12/2018-07-12_13-18-00.jpg'
    motion_test_4 = '../sky_image/2018-07-12/2018-07-12_13-18-10.jpg'
    motion_test_5 = '../sky_image/2018-07-04/2018-07-04_11-56-30.jpg'
    motion_test_6 = '../sky_image/2018-07-04/2018-07-04_11-56-40.jpg'
    half_covered = '../sky_image/2018-07-04/2018-07-04_12-31-20.jpg'
    full_covered = '../sky_image/2018-07-04/2018-07-04_12-32-40.jpg'
    sun_rise = "../sky_image/2018-07-04/2018-07-04_06-50-00.jpg"
    edge_clear = "../sky_image/2018-07-04/2018-07-04_15-18-00.jpg"
    edge_cloudy = "../sky_image/2018-07-04/2018-07-04_08-51-20.jpg"
    very_sink = "../sky_image/2018-07-09/2018-07-09_13-35-00.jpg"
    changeable = '../sky_image/2018-07-01/2018-07-01_08-15-00.jpg'

    start = time.time()
    image1 = SkyImage(motion_test_2)
    # image1.show_brighter()
    # image1.show_minmax_brbg()
    image1.show_threshold()
    # image1.save_threshold_brbg("../img1.jpg")
    # image1.show_threshold_minmax()
    end = time.time()
    print(end-start)
    # plt.hist(image1.brbg_ratio().reshape(-1), bins=40)
    # plt.show()
    # np.savetxt("../brbg.csv", image1.brbg_ratio(), delimiter=",")
    # image1.save_threshold_brbg("../img1.jpg")
    # image1.kmeans(image1.minmax_brbg_ratio(), SEC_CROP)
    # image1.show_resized()

    # image2 = SkyImage(motion_test_4)
    # image2.save_threshold_brbg("../img2.jpg")



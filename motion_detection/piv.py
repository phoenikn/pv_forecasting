import time

import numpy
from openpiv import tools, pyprocess, validation, filters, scaling
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import cloud_detection.cloud_detection as cd

WIN_SIZE = 32
SEARCH_SIZE = 38
OVERLAP = 12
DT = 0.02
# 364 / 13 = 28
PIV_RESIZE_RATIO = 28


class PivDetector:

    def __init__(self, image1_dir: str, image2_dir: str):
        self.image1 = cd.SkyImage(image1_dir)
        self.image2 = cd.SkyImage(image2_dir)
        self.cloud_1 = np.asarray(self.image1.get_image_gray())
        self.cloud_2 = np.asarray(self.image2.get_image_gray())
        # fig, ax = plt.subplots(1, 2, figsize=(12, 10))
        # ax[0].imshow(self.cloud_1, cmap=plt.cm.gray)
        # ax[1].imshow(self.cloud_2, cmap=plt.cm.gray)

    def piv(self):
        u0, v0, sig2noise = pyprocess.extended_search_area_piv(self.cloud_1.astype(np.int32),
                                                               self.cloud_2.astype(np.int32),
                                                               window_size=WIN_SIZE,
                                                               overlap=OVERLAP,
                                                               dt=DT,
                                                               search_area_size=SEARCH_SIZE,
                                                               sig2noise_method='peak2peak')

        x, y = pyprocess.get_coordinates(image_size=self.cloud_1.shape,
                                         search_area_size=SEARCH_SIZE,
                                         overlap=OVERLAP)

        # Used to delete noises (spurious vectors)
        u1, v1, mask = validation.sig2noise_val(u0, v0, sig2noise, threshold=1.05)

        u2, v2 = filters.replace_outliers(u1, v1, method='localmean', max_iter=3, kernel_size=3)

        x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor=100)  #  microns/pixel

        # 0,0 shall be bottom left, positive rotation rate is counterclockwise
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not mask[i][j]:
                    u3[i][j] = 0.0
                    v3[i][j] = 0.0

        # tools.save(x, y, u3, v3, mask, 'motion.txt')
        # fig, ax = plt.subplots(figsize=(8, 8))
        # tools.display_vector_field('motion.txt',
        #                            ax=ax, scaling_factor=100,
        #                            scale=50,  # scale defines here the arrow length
        #                            width=0.0035,  # width is the thickness of the arrow
        #                            on_img=True,  # overlay on the image
        #                            image_name='../img1.jpg')

        u3 = np.repeat(u3, PIV_RESIZE_RATIO, axis=1)
        u3 = np.repeat(u3, PIV_RESIZE_RATIO, axis=0)
        v3 = np.repeat(v3, PIV_RESIZE_RATIO, axis=1)
        v3 = np.repeat(v3, PIV_RESIZE_RATIO, axis=0)

        return u3, v3

    def get_cloud_255(self):
        return self.cloud_2

    def get_cloud_bi(self):
        return 1 - self.image2.image_process(cd.ImageChannelProcess.BRBG)

    def get_image_gray(self):
        return np.asarray(self.image2.get_image_gray())

    def get_sun_matrix(self):
        radius = round(80 * 512 / 1551)
        img_size = self.get_cloud_bi().shape
        sun_center = self.image2.get_sun_center()
        sun_center_x, sun_center_y = sun_center
        background = Image.new("L", img_size)
        draw = ImageDraw.Draw(background)
        left_up = (sun_center_x - radius, sun_center_y - radius)
        right_down = (sun_center_x + radius, sun_center_y + radius)
        points = [left_up, right_down]
        draw.ellipse(points, fill=1)
        background = np.asarray(background)
        return background


if __name__ == "__main__":
    motion_test_1 = '../sky_image/2018-07-04/2018-07-04_12-08-40.jpg'
    motion_test_2 = '../sky_image/2018-07-04/2018-07-04_12-08-30.jpg'
    motion_test_3 = '../sky_image/2018-07-12/2018-07-12_13-18-00.jpg'
    motion_test_4 = '../sky_image/2018-07-12/2018-07-12_13-18-10.jpg'
    motion_test_5 = '../sky_image/2018-07-04/2018-07-04_11-56-30.jpg'
    motion_test_6 = '../sky_image/2018-07-04/2018-07-04_11-56-40.jpg'
    motion_test_7 = '../sky_image/2018-07-04/2018-07-04_08-05-40.jpg'
    motion_test_8 = '../sky_image/2018-07-04/2018-07-04_08-05-50.jpg'

    piv = PivDetector(motion_test_1, motion_test_2)
    piv.piv()
    # cloud = piv.get_cloud_bi()
    # plt.imshow(piv.get_sun_matrix(), cmap="gray")
    # plt.show()
    # print(cloud.shape, velocity1.shape, velocity2.shape)

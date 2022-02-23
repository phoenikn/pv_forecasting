import openpiv.piv
from openpiv import tools, pyprocess, validation, filters, scaling

import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

# img1 = Image.open('sky_image/2018-07-04/2018-07-04_12-08-40.jpg')
# img2 = Image.open('sky_image/2018-07-04/2018-07-04_12-08-30.jpg')
# img1 = img1.crop((256, 0, 1792, 1536))
# img1 = img1.resize((512, 512))
# img1 = img1.crop((75, 75, 437, 437))
# img2 = img2.crop((256, 0, 1792, 1536))
# img2 = img2.resize((512, 512))
# img2 = img2.crop((75, 75, 437, 437))
# img1.save("img1.jpg")
# img2.save("img2.jpg")

start = time.time()

frame_a = tools.imread('img1.jpg')
frame_b = tools.imread('img2.jpg')
fig, ax = plt.subplots(1, 2, figsize=(12, 10))
ax[0].imshow(frame_a, cmap=plt.cm.gray)
ax[1].imshow(frame_b, cmap=plt.cm.gray)

img1 = Image.open('img1.jpg').convert("L")

winsize = 32
searchsize = 38
overlap = 12
dt = 0.02

u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32),
                                                       frame_b.astype(np.int32),
                                                       window_size=winsize,
                                                       overlap=overlap,
                                                       dt=dt,
                                                       search_area_size=searchsize,
                                                       sig2noise_method='peak2peak')

x, y = pyprocess.get_coordinates(image_size=frame_a.shape,
                                 search_area_size=searchsize,
                                 overlap=overlap)

# Used to delete noises (spurious vectors)
u1, v1, mask = validation.sig2noise_val(u0, v0,
                                        sig2noise,
                                        threshold=1.05)

u2, v2 = filters.replace_outliers(u1, v1,
                                  method='localmean',
                                  max_iter=3,
                                  kernel_size=3)

x, y, u3, v3 = scaling.uniform(x, y, u2, v2,
                               scaling_factor=100)  # 96.52 microns/pixel

# 0,0 shall be bottom left, positive rotation rate is counterclockwise
x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)


for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if not mask[i][j]:
            u3[i][j] = 0.0
            v3[i][j] = 0.0

# save in the simple ASCII table format
# tools.save(x, y, u3, v3, mask, 'motion.txt')
# fig, ax = plt.subplots(figsize=(8, 8))
# tools.display_vector_field('motion.txt',
#                            ax=ax, scaling_factor=100,
#                            scale=50,  # scale defines here the arrow length
#                            width=0.0035,  # width is the thickness of the arrow
#                            on_img=True,  # overlay on the image
#                            image_name='img1.jpg')
u3 = np.repeat(u3, 28, axis=1)
u3 = np.repeat(u3, 28, axis=0)
v3 = np.repeat(v3, 28, axis=1)
v3 = np.repeat(v3, 28, axis=0)

print(u3)
print(v3)
#
# end = time.time()
# print(end-start)

# im1 = frame_a
# im2 = frame_b
#
# u, v, s2n = pyprocess.extended_search_area_piv(
#     im1.astype(np.int32), im2.astype(np.int32), window_size=32,
#     overlap=16, search_area_size=32
# )
# x, y = pyprocess.get_coordinates(image_size=im1.shape,
#                                  search_area_size=32, overlap=16)
#
# valid = s2n > np.percentile(s2n, 5)
#
# _, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(im1, cmap=plt.get_cmap("gray"), alpha=0.5, origin="upper")
# ax.quiver(x[valid], y[valid], u[valid], -v[valid], scale=70,
#           color='r', width=.005)
# plt.show()
# print(valid)

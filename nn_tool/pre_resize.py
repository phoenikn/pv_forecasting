import os
import time
from time import time as tm
from os.path import join as path_join

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

ABSOLUTE_IMG_DIR_1 = "/scratch/itee/uqsxu13/2020_data/2020_gatton_1"
PRE_DIR = "/scratch/itee/uqsxu13/2020_data/pre_compress/"


transform = transforms.Compose([
                transforms.CenterCrop((1500, 1500)),
                transforms.Resize((256, 256))
            ])

days = os.listdir(ABSOLUTE_IMG_DIR_1)
for day in days:
    day_dir = os.path.join(ABSOLUTE_IMG_DIR_1, day)
    files = os.listdir(day_dir)
    save_dir = os.path.join(PRE_DIR, day)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    amount = len(files)

    for i, file in enumerate(files, 1):
        img_dir = os.path.join(day_dir, file)
        img = Image.open(img_dir)
        img = transform(img)
        img.save(os.path.join(save_dir, file))
        print(i/amount)


# for file in files:
#     print(file)

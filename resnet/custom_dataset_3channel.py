import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from custom_dataset import PvImageFeatureDataset


class PvImage3ChannelFeatureDataset(PvImageFeatureDataset):

    def __getitem__(self, index):
        time = self.index_df["second"][index]
        label = self.index_df["label"][index]

        if label == 1:
            feature_folder = self.cliff_folder
        elif label == 0:
            feature_folder = self.ordinary_folder
        else:
            raise Exception("Wrong label")

        feature_zip = np.load(os.path.join(feature_folder, time + ".npz"))
        stack = torch.empty([0, 224, 224])
        sun_matrix, cloud, velocity1, velocity2 = feature_zip
        features = [feature_zip[sun_matrix],
                    np.multiply(feature_zip[cloud], feature_zip[velocity1]),
                    np.multiply(feature_zip[cloud], feature_zip[velocity2])]

        if self.transform is not None:
            for feature in features:
                r = self.transform(feature)
                stack = torch.cat((stack, r), 0)
                if stack.size()[0] > 3:
                    raise Exception("More feature than expected")
        else:
            raise Exception("No transformer")

        if self.transform_extra is not None:
            stack = self.transform_extra(stack)

        return stack, label

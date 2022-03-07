import os
import torch
import torch.utils.data
import numpy as np

from custom_dataset import PvImageFeatureDataset


class PvImage4ChannelFeatureDataset(PvImageFeatureDataset):
    """The four layers are: sun, cloud v1, cloud v2, gray image"""

    def __getitem__(self, index):
        mean = False
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
        sun_matrix, cloud, velocity1, velocity2, gray_img = feature_zip.values()

        if mean:
            velocity1 = np.full(velocity1.shape, velocity1.mean())
            velocity2 = np.full(velocity2.shape, velocity2.mean())

        features = [sun_matrix,
                    np.multiply(cloud, velocity1),
                    np.multiply(cloud, velocity2),
                    gray_img]

        if self.transform is not None:
            for feature in features:
                r = self.transform(feature)
                stack = torch.cat((stack, r), 0)
                if stack.size()[0] > 4:
                    raise Exception("More feature than expected")
        else:
            raise Exception("No transformer")

        if self.transform_extra is not None:
            stack = self.transform_extra(stack)

        return stack, label

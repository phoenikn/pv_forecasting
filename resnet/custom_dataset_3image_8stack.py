import os
import torch
import torch.utils.data
import numpy as np

from custom_dataset import PvImageFeatureDataset


class PvImage8ChannelFeatureDataset(PvImageFeatureDataset):
    """The eight layers are: sun1, cloud v1, cloud v2, gray image1, sun2, cloud v3, cloud v4, gray image2"""

    def __getitem__(self, index):
        mean = False
        time = self.index_df["third"][index]
        label = self.index_df["label"][index]

        if label == 1:
            feature_folder = self.cliff_folder
        elif label == 0:
            feature_folder = self.ordinary_folder
        else:
            raise Exception("Wrong label")

        feature_zip = np.load(os.path.join(feature_folder, time + ".npz"))
        stack = torch.empty([0, 224, 224])
        sun_matrix1, cloud1, velocity1, velocity2, gray_img1,\
            sun_matrix2, cloud2, velocity3, velocity4, gray_img2 = feature_zip.values()

        if mean:
            velocity1 = np.full(velocity1.shape, velocity1.mean())
            velocity2 = np.full(velocity2.shape, velocity2.mean())
            velocity3 = np.full(velocity3.shape, velocity3.mean())
            velocity4 = np.full(velocity4.shape, velocity4.mean())

        features = [sun_matrix1,
                    np.multiply(cloud1, velocity1),
                    np.multiply(cloud1, velocity2),
                    gray_img1,
                    sun_matrix2,
                    np.multiply(cloud2, velocity3),
                    np.multiply(cloud2, velocity4),
                    gray_img2]

        if self.transform is not None:
            for feature in features:
                r = self.transform(feature)
                stack = torch.cat((stack, r), 0)
                if stack.size()[0] > 8:
                    raise Exception("More feature than expected")
        else:
            raise Exception("No transformer")

        if self.transform_extra is not None:
            stack = self.transform_extra(stack)

        return stack, label

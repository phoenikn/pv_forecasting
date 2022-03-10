import os
import torch
import torch.utils.data
import numpy as np

from custom_dataset import PvImageFeatureDataset


class PvImage2ChannelNumericDataset(PvImageFeatureDataset):
    """The eight layers are: sun1, cloud v1, cloud v2, gray image1, sun2, cloud v3, cloud v4, gray image2"""

    def __getitem__(self, index):
        """label is the difference between now output and 30s later output, used for regression"""
        time = self.index_df["third"][index]
        label = self.index_df["difference"][index]
        feature_folder = self.cliff_folder

        feature_zip = np.load(os.path.join(feature_folder, time + ".npz"))
        stack = torch.empty([0, 224, 224])
        sun_matrix1, cloud1, velocity1, velocity2, gray_img1,\
            sun_matrix2, cloud2, velocity3, velocity4, gray_img2 = feature_zip.values()

        features = [gray_img1, gray_img2]

        if self.transform is not None:
            for feature in features:
                r = self.transform(feature)
                stack = torch.cat((stack, r), 0)
                if stack.size()[0] > 2:
                    raise Exception("More feature than expected")
        else:
            raise Exception("No transformer")

        if self.transform_extra is not None:
            stack = self.transform_extra(stack)

        return stack, label

import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torchvision import transforms
import pandas as pd
import numpy as np


class PvImageFeatureDataset(torch.utils.data.Dataset):
    """Able to customize the import features"""

    def __init__(self, index_path, file_folder, channels, transform=None, transform_extra=None,
                 tensor_size=(224, 224), select_feature: list = None):
        self.index_df = pd.read_csv(index_path)
        self.file_folder = file_folder
        self.transform = transform
        self.transform_extra = transform_extra
        self.channels = channels
        self.tensor_size = tensor_size
        self.select_feature = select_feature

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, index):
        column_names = list(self.index_df)
        time = self.index_df[column_names[-2]][index]
        label = self.index_df[column_names[-1]][index]

        feature_zip = np.load(os.path.join(self.file_folder, time + ".npz"))
        stack = torch.empty([0, self.tensor_size[0], self.tensor_size[1]])

        if self.transform is not None:
            for feature in feature_zip:
                if self.select_feature is not None:
                    if feature in self.select_feature:
                        r = self.transform(feature_zip[feature])
                        stack = torch.cat((stack, r), 0)
                else:
                    r = self.transform(feature_zip[feature])
                    stack = torch.cat((stack, r), 0)
                if stack.size()[0] > self.channels:
                    raise Exception("More feature than expected")
        else:
            raise Exception("No transformer")

        if self.transform_extra is not None:
            stack = self.transform_extra(stack)

        stack = stack[:, None, :, :]
        return stack, label


if __name__ == "__main__":
    index_dir = "../index/cliff_time_points_three_numeric.csv"
    file_dir = "../extracted_data/test"

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224)),
    # ])
    #
    # one = PvImageFeatureDataset(index_dir, file_dir, channels=3, transform=transform_test)
    # plt.imshow(one[0].__getitem__(0)[0], cmap="gray")
    # plt.show()

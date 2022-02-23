import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np


class PvImageFeatureDataset(torch.utils.data.Dataset):

    def __init__(self, index_path, ordinary_folder, cliff_folder, transform=None, transform_extra=None):
        self.index_df = pd.read_csv(index_path)
        self.ordinary_folder = ordinary_folder
        self.cliff_folder = cliff_folder
        self.transform = transform
        self.transform_extra = transform_extra

    def __len__(self):
        return len(self.index_df)

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

        if self.transform is not None:
            for feature in feature_zip:
                r = self.transform(feature_zip[feature])
                stack = torch.cat((stack, r), 0)
                if stack.size()[0] > 4:
                    raise Exception("More feature than expected")
        else:
            raise Exception("No transformer")

        if self.transform_extra is not None:
            stack = self.transform_extra(stack)

        return stack, label


if __name__ == "__main__":
    index_dir = "../index/combined_ordinary_cliff.csv"
    ordinary = "../extracted_1min_before_cliff"
    cliff = "../extracted_cliff"
    #
    # one = PvImageFeatureDataset(index_dir, ordinary, cliff, transform=test_transform)
    # print(one.__getitem__(0))

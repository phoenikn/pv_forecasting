import os

import pandas as pd
from typing import List
from pv_data_process import DATA_IMG_VAL

DATA_FOLDER = "../pv_data"
DATA_HAVE_IMG = "data_has_img.csv"


class RawCsvFile:

    def __init__(self, raw_data_name_, data_folder: str = DATA_FOLDER):
        self._raw = pd.read_csv(os.path.join(data_folder, raw_data_name_))

    def _month_split(self) -> List[pd.DataFrame]:
        july_data = self._raw[self._raw["DateTime"].str.startswith("2018-07")]
        august_data = self._raw[self._raw["DateTime"].str.startswith("2018-08")]
        september_data = self._raw[self._raw["DateTime"].str.startswith("2018-09")]

        return [july_data, august_data, september_data]

    @staticmethod
    def df_to_files(data_list: List[pd.DataFrame], file_names_: List[str], folder_dir: str) -> None:
        for i in range(0, len(data_list)):
            data_list[i].to_csv(os.path.join(folder_dir, file_names_[i] + ".csv"), index=False)

    def month_split_to_file(self, names: List[str]):
        self.df_to_files(self._month_split(), names, DATA_FOLDER)


if __name__ == "__main__":
    file_names = ["data_img_val_2018-07", "data_img_val_2018-08", "data_img_val_2018-09"]
    # raw = RawCsvFile(os.path.join(DATA_FOLDER, DATA_HAVE_IMG))
    # raw.month_split_to_file(file_names)
    # data_img_val = RawCsvFile(DATA_IMG_VAL)
    # data_img_val.month_split_to_file(file_names)

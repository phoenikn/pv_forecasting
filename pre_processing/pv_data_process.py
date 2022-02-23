import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

DATE_TIME_COLUMN = "DateTime"
POWER_COLUMN = "Power(kW)"
IMAGE_NAME_COLUMN = "img_name"

DATA_FOLDER = "../pv_data"
INDEX_FOLDER = "../index"

DATA_RAW_ALL = "raw_all_data.csv"
DATA_HAVE_IMG = "data_has_img.csv"
DATA_JULY = "2018-07.csv"
DATA_AUGUST = "2018-08.csv"
DATA_SEP = "2018-09.csv"
DATA_ALL_NORM = "all_norm.csv"
DATA_ALL_WITH_ZERO = "all_with_zero.csv"
DATA_IMG_VAL = "data_has_img_and_value.csv"
DATA_IMG_VAL_FILES = {"july": "data_img_val_2018-07.csv",
                      "august": "data_img_val_2018-08.csv",
                      "september": "data_img_val_2018-09.csv"}

IMG_ALL_NAMES = "img_names.csv"

ZERO_THRESHOLD = 0.003


class PvData:

    def __init__(self, raw_data_name_, data_folder: str = DATA_FOLDER, preprocessed: bool = False) -> None:
        self._raw_data = pd.read_csv(os.path.join(data_folder, raw_data_name_))
        self._preprocessed = preprocessed
        if not self._preprocessed:
            self._positive_power = self.get_positive_power()
            self._min_max = MinMaxScaler()
            self._min_max.fit(self._positive_power.to_frame())

    def get_data_df(self) -> pd.DataFrame:
        return self._raw_data

    def get_powers(self) -> pd.Series:
        return self._raw_data[POWER_COLUMN]

    def get_datetime(self) -> pd.Series:
        return self._raw_data[DATE_TIME_COLUMN]

    def get_positive_power(self) -> pd.Series:
        return self.get_powers() * -1

    def get_norm_min_max(self) -> pd.Series:
        if self._preprocessed:
            return self.get_powers()
        else:
            return pd.Series(self._min_max.transform(self._positive_power.to_frame())[:, 0],
                             name=self._positive_power.name)

    @staticmethod
    def plot_data(data: pd.Series, title: str, days: int = -1) -> None:
        data.plot()
        plt.title(title)
        if days != -1:
            length = data.size
            plt.xticks(np.linspace(0, length - 1, days + 1)[:-1], range(1, days + 1))
        plt.show()

    def plot_norm_all(self) -> None:
        self.plot_data(self.get_norm_min_max(), "Min-max Normalization")

    def plot_norm_days(self, days: int) -> None:
        self.plot_data(self.get_norm_min_max(), "Min-max Normalization {}days".format(days), days)

    @staticmethod
    def save_data(data: pd.DataFrame, name: str) -> None:
        data.to_csv(os.path.join(DATA_FOLDER, name + ".csv"), index=False)

    def plot_one_day(self, month_: int, day_: int):
        month_ = "{:02d}".format(month_)
        day_ = "{:02d}".format(day_)
        months = self.get_datetime().apply(lambda x: x.split("_")[0][5:7])
        days = self.get_datetime().apply(lambda x: x.split("_")[0][-2:])
        powers_of_day = self.get_powers()[days == day_]
        powers_of_day = powers_of_day[months == month_]
        self.plot_data(powers_of_day, "Power in 2018-{}-{}".format(month_, day_))

    def plot_short_series(self, start_index: int, end_index: int):
        self.plot_data(self.get_powers()[start_index:end_index],
                       "Power from index {} to {}".format(start_index, end_index))


class NormPvData(PvData):

    def __init__(self, raw_data_name_, data_folder: str = DATA_FOLDER, preprocessed: bool = True):
        super().__init__(raw_data_name_, data_folder, preprocessed)

    def set_zero(self) -> pd.Series:
        return self.get_powers().apply(lambda x: 0 if x < ZERO_THRESHOLD else x)

    def plot_norm_zero_all(self) -> None:
        self.plot_data(self.set_zero(), "Normalized data with zero")

    def save_zero(self) -> None:
        self.save_data(pd.concat([self.get_datetime(), self.set_zero()], axis=1), "all_with_zero")

    def select_non_zero_time_duplicate(self) -> pd.Series:
        data = self.get_data_df()
        rows_has_values = data[data[POWER_COLUMN] != 0.0]
        time_10s_has_values_duplicate = rows_has_values[DATE_TIME_COLUMN].apply(lambda x: x[:-1])
        return time_10s_has_values_duplicate

    def select_non_zero_data(self) -> pd.DataFrame:
        time_has_value_10s = self.get_datetime().apply(lambda x: x[:-1])
        data_has_value = self.get_data_df()[time_has_value_10s.isin(self.select_non_zero_time_duplicate())]
        return data_has_value

    def select_zero_data(self) -> pd.DataFrame:
        time_no_value_10s = self.get_datetime().apply(lambda x: x[:-1])
        data_no_value = self.get_data_df()[~time_no_value_10s.isin(self.select_non_zero_time_duplicate())]
        return data_no_value

    def save_data_with_value(self) -> None:
        self.save_data(self.select_non_zero_data(), "data_has_img_and_value")

    def save_data_no_value(self) -> None:
        self.save_data(self.select_zero_data(), "data_has_no_value")


class ImageIndex:

    def __init__(self, img_index_name_, index_folder: str = INDEX_FOLDER) -> None:
        self._img_index = pd.read_csv(os.path.join(index_folder, img_index_name_))

    def get_index(self) -> pd.DataFrame:
        return self._img_index

    def select_img_by_time(self, time_10s_has_values: pd.Series) -> pd.DataFrame:
        img_name_10s = self.get_index()[IMAGE_NAME_COLUMN].apply(lambda x: x[:-1])
        img_has_value = self.get_index()[img_name_10s.isin(time_10s_has_values)]
        return img_has_value

    def save_img_has_value(self, time_has_value: pd.Series):
        self.save_index(self.select_img_by_time(time_has_value),
                        "img_has_value")

    @staticmethod
    def save_index(index: pd.DataFrame, name: str) -> None:
        index.to_csv(os.path.join(INDEX_FOLDER, name + ".csv"), index=False)


if __name__ == "__main__":
    # all_data = PvData(DATA_RAW_ALL)
    # all_data.plot_data(all_data.get_powers(), "Raw data")
    # all_norm = NormPvData(DATA_ALL_NORM)
    # all_norm.save_zero()
    # data_all_with_zero = NormPvData(DATA_ALL_WITH_ZERO)
    # image_index_all = ImageIndex(IMG_ALL_NAMES)
    # data_all_with_zero.save_data_with_value()
    # data_all_with_zero.save_data_no_value()
    # image_index_all.save_img_has_value(data_all_with_zero.select_non_zero_time_duplicate())
    # data_has_img_and_value = NormPvData(DATA_IMG_VAL)
    data_img_val_july = NormPvData(DATA_IMG_VAL_FILES["august"])
    # data_img_val_aug = NormPvData(DATA_IMG_VAL_FILES["august"])
    data_img_val_july.plot_one_day(8, 3)
    # for day in range(1, 16):
    #     data_img_val_july.plot_one_day(7, day)
    # data_img_val_july.plot_short_series(690000, 691000)

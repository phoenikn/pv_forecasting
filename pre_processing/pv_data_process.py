import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

DATE_TIME_COLUMN = "DateTime"
POWER_COLUMN = "Power(kW)"
IMAGE_NAME_COLUMN = "img_name"

ABSOLUTE_FILE_DIR = "C:/Users/s4544852/Desktop/gatton PV data/Data for CSIRO/2020"
ABSOLUTE_IMG_DIR_1 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 1/2020"
ABSOLUTE_IMG_DIR_2 = "C:/Users/s4544852/Desktop/gatton PV data/Gatton 2/2020"
ABSOLUTE_INDEX_DIR = "C:/Users/s4544852/Desktop/gatton PV data/index_2020"

DATA_FOLDER = "../pv_data"
INDEX_FOLDER = "../index"

# DATA_RAW_ALL = "raw_all_data.csv"
# DATA_HAVE_IMG = "data_has_img.csv"
# DATA_JULY = "2018-07.csv"
# DATA_AUGUST = "2018-08.csv"
# DATA_SEP = "2018-09.csv"
# DATA_ALL_NORM = "all_norm.csv"
# DATA_ALL_WITH_ZERO = "all_with_zero.csv"
# DATA_IMG_VAL = "data_has_img_and_value.csv"
# DATA_IMG_VAL_FILES = {"july": "data_img_val_2018-07.csv",
#                       "august": "data_img_val_2018-08.csv",
#                       "september": "data_img_val_2018-09.csv"}

DATA_RAW_ALL = "raw_all_data.csv"
DATA_HAVE_IMG = "data_has_img.csv"
DATA_JULY = "2018-07.csv"
DATA_AUGUST = "2018-08.csv"
DATA_SEP = "2018-09.csv"
DATA_ALL_NORM = "all_norm.csv"
DATA_ALL_WITH_ZERO = "all_with_zero.csv"
DATA_IMG_VAL = "data_has_img_and_value.csv"

IMG_ALL_NAMES = "img_names.csv"

ZERO_THRESHOLD = 0.003


class PvData:

    def __init__(self, raw_data_name_, data_folder: str = DATA_FOLDER, preprocessed: bool = False) -> None:
        print(os.path.join(data_folder, raw_data_name_))
        self._raw_data = pd.read_csv(os.path.join(data_folder, raw_data_name_))
        self._preprocessed = preprocessed
        if not self._preprocessed:
            if self.get_positive_power().dtype != np.float64:
                self.null_clean()
            self._positive_power = self.get_positive_power()
            self._min_max = MinMaxScaler()
            self._min_max.fit(self._positive_power.to_frame())

    def get_data_df(self) -> pd.DataFrame:
        return self._raw_data

    def null_clean(self):
        self._raw_data[POWER_COLUMN].replace(to_replace="(null)", method="ffill", inplace=True)
        self._raw_data[POWER_COLUMN] = pd.to_numeric(self.get_powers())

    def get_powers(self) -> pd.Series:
        return self._raw_data[POWER_COLUMN]

    def get_datetime(self) -> pd.Series:
        return self._raw_data[DATE_TIME_COLUMN]

    def get_positive_power(self) -> pd.Series:
        return self.get_powers().apply(lambda x: x * -1)

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

    def save_processed(self, name: str, data_folder: str = DATA_FOLDER):
        processed = pd.concat([self.get_datetime(), self.get_norm_min_max()],
                              axis=1)
        self.save_data(processed, name, data_folder)

    @staticmethod
    def save_data(data: pd.DataFrame, name: str, data_folder: str = DATA_FOLDER) -> None:
        data.to_csv(os.path.join(data_folder, name + ".csv"), index=False)

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

    def save_zero(self, name: str = "all_with_zero", data_folder: str = DATA_FOLDER) -> None:
        self.save_data(pd.concat([self.get_datetime(), self.set_zero()], axis=1), name=name,
                       data_folder=data_folder)

    def select_non_zero_time_duplicate(self) -> pd.Series:
        data = self.get_data_df()
        rows_has_values = data[data[POWER_COLUMN] != 0.0]
        time_10s_has_values_duplicate = rows_has_values[DATE_TIME_COLUMN].apply(lambda x: x[:-1])
        return time_10s_has_values_duplicate

    def select_non_zero_time_from_value(self, has_zero=False) -> pd.Series:
        if not has_zero:
            return self.get_datetime().apply(lambda x: x[:-3])

    def select_non_zero_data(self) -> pd.DataFrame:
        time_has_value_10s = self.get_datetime().apply(lambda x: x[:-1])
        data_has_value = self.get_data_df()[time_has_value_10s.isin(self.select_non_zero_time_duplicate())]
        return data_has_value

    def select_non_zero_1min(self) -> pd.DataFrame:
        data_has_value = self.get_data_df()[self.get_powers() != 0.0]
        return data_has_value

    def select_zero_data(self) -> pd.DataFrame:
        time_no_value_10s = self.get_datetime().apply(lambda x: x[:-1])
        data_no_value = self.get_data_df()[~time_no_value_10s.isin(self.select_non_zero_time_duplicate())]
        return data_no_value

    def save_data_with_value(self, one_min_res=False, name: str = "data_has_value",
                             data_folder: str = DATA_FOLDER) -> None:
        if one_min_res:
            self.save_data(self.select_non_zero_1min(), name, data_folder)
        else:
            self.save_data(self.select_non_zero_data(), name, data_folder)

    def save_data_no_value(self) -> None:
        self.save_data(self.select_zero_data(), "data_has_no_value")

    def select_save_data_with_val_img(self, min_with_img_val_, data_folder, index_folder, name="data_2020", save=False):
        time_column = self.select_non_zero_time_from_value()
        data_with_val_img = self.get_data_df()[time_column.isin(min_with_img_val_)]
        if save:
            data_with_val_img.to_csv(os.path.join(index_folder, name + ".csv"), index=False)
            data_with_val_img.to_csv(os.path.join(data_folder, name + ".csv"), index=False)
        return data_with_val_img



class ImageIndex:

    def __init__(self, img_index_name_, index_folder: str = INDEX_FOLDER) -> None:
        self._img_index = pd.read_csv(os.path.join(index_folder, img_index_name_))

    def get_index(self) -> pd.DataFrame:
        return self._img_index

    def select_img_by_time(self, time_10s_has_values: pd.Series) -> pd.DataFrame:
        img_name_10s = self.get_index()[IMAGE_NAME_COLUMN].apply(lambda x: x[:-1])
        img_has_value = self.get_index()[img_name_10s.isin(time_10s_has_values)]
        return img_has_value

    def select_save_img_by_time_min(self, time_min_has_values: pd.Series,
                                    index_folder=ABSOLUTE_INDEX_DIR, save=False) -> pd.Series:
        img_name_min = self.get_index()[IMAGE_NAME_COLUMN].apply(lambda x: x[:-3])
        img_has_six = img_name_min.value_counts()[img_name_min.value_counts() == 6].index.to_series()\
            .reset_index(drop=True)
        img_has_six_and_value_min = pd.Series(np.intersect1d(img_has_six.values, time_min_has_values.values))
        if save:
            img_has_six_and_value = self.get_index()[img_name_min.isin(img_has_six_and_value_min)]
            img_has_six_and_value.to_csv(os.path.join(index_folder, "img_has_value_2020.csv"), index=False)
        return img_has_six_and_value_min

    def save_img_has_value(self, time_has_value: pd.Series, index_folder: str = INDEX_FOLDER):
        self.save_index(self.select_img_by_time(time_has_value),
                        "img_has_value", index_folder)

    @staticmethod
    def save_index(index: pd.DataFrame, name: str, index_folder: str = INDEX_FOLDER) -> None:
        index.to_csv(os.path.join(index_folder, name + ".csv"), index=False)


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
    # data_img_val_july = NormPvData(DATA_IMG_VAL_FILES["july"])
    # data_img_val_aug = NormPvData(DATA_IMG_VAL_FILES["august"])
    # data_img_val_july.plot_one_day(7, 4)
    # for day in range(1, 16):
    #     data_img_val_july.plot_one_day(7, day)
    # data_img_val_july.plot_short_series(690000, 691000)

    # pv_data_2020 = PvData("pv_2020.csv", data_folder=ABSOLUTE_FILE_DIR)
    # pv_data_2020.save_processed(name="pv_2020_norm", data_folder=ABSOLUTE_FILE_DIR)

    # norm_data_2020 = NormPvData("pv_2020_norm.csv", data_folder=ABSOLUTE_FILE_DIR)
    # norm_data_2020.save_zero(name="pv_2020_zero", data_folder=ABSOLUTE_FILE_DIR)

    # zero_data_2020 = NormPvData("pv_2020_zero.csv", data_folder=ABSOLUTE_FILE_DIR)
    # zero_data_2020.save_data_with_value(True, name="pv_2020_has_value", data_folder=ABSOLUTE_FILE_DIR)

    # from handle_csv import csv_time_format_change
    # csv_time_format_change(os.path.join(ABSOLUTE_FILE_DIR, "pv_2020_has_value.csv"), ABSOLUTE_FILE_DIR,
    #                        name="pv_2020_has_value_formatted.csv")

    image_index_all = ImageIndex("img_names_2020.csv", index_folder=ABSOLUTE_INDEX_DIR)
    data_all_value = NormPvData("pv_2020_has_value_formatted.csv", data_folder=ABSOLUTE_FILE_DIR)
    min_with_img_val = image_index_all.select_save_img_by_time_min(data_all_value.select_non_zero_time_from_value())
    data_all_value.select_save_data_with_val_img(min_with_img_val, ABSOLUTE_FILE_DIR, ABSOLUTE_INDEX_DIR)
    # image_index_all.save_img_has_value(data_all_value.select_non_zero_time_duplicate())


import pandas as pd
from datetime import datetime
from tools.util import *
from matplotlib import pyplot as plt


def concat_pv_csv(data_directory_):
    csv_out = None
    for filename in os.listdir(data_directory_):
        pv_csv = pd.read_csv(os.path.join(data_directory_, filename))
        if csv_out is None:
            csv_out = pv_csv
        else:
            pv_csv = pv_csv[~pv_csv["DateTime"].isin(csv_out["DateTime"])]
            csv_out = pd.concat([csv_out, pv_csv])

    csv_out.rename(columns={"AQG1_B001_PM001.Sts.P_kW": "Power(kW)"}, inplace=True)
    csv_out.to_csv(os.path.join(data_directory_, "raw_all_data.csv"), index=False)


def select_data_with_img(img_names_dir_, formatted_data_dir_, data_dir_):
    img_names_ = pd.read_csv(img_names_dir_)["img_name"]
    raw_data_ = pd.read_csv(formatted_data_dir_)
    img_names_10s = img_names_.apply(lambda x: x[:-1])
    data_has_img = raw_data_[raw_data_["DateTime"].str[:-1].isin(img_names_10s)]
    data_has_img.to_csv(os.path.join(data_dir_, "data_has_img.csv"), index=False)


def select_data_without_img(img_names_dir_, formatted_data_dir_, data_dir_):
    img_names_ = pd.read_csv(img_names_dir_)["img_name"]
    raw_data_ = pd.read_csv(formatted_data_dir_)
    img_names_10s = img_names_.apply(lambda x: x[:-1])
    data_has_img = raw_data_[~raw_data_["DateTime"].str[:-1].isin(img_names_10s)]
    data_has_img.to_csv(os.path.join(data_dir_, "data_has_no_img.csv"), index=False)


def extract_img_names(img_directory_, other_data_dir_, name: str = "img_names.csv"):
    img_names_ = pd.Series(dtype="string", name="img_name")
    for folder_name in os.listdir(img_directory_):
        if os.path.isdir(os.path.join(img_directory_, folder_name)):
            img_names_ = img_names_.append(pd.Series(os.listdir(os.path.join(img_directory_, folder_name))),
                                           ignore_index=True)

    img_names_ = img_names_[img_names_.str.endswith(".jpg")]
    img_names_ = img_names_.apply(lambda x: x[:-4])
    img_names_ = img_names_.to_frame()
    img_names_.columns = ["img_name"]
    print(img_names_[~img_names_["img_name"].str.endswith("0")])
    # img_names_.to_csv(os.path.join(other_data_dir_, name), index=False)


def round_img_name(img_names_dir_, img_dir_, data_dir_):
    """
    Only run once
    :param data_dir_:
    :param img_names_dir_:
    :param img_dir_:
    :return:
    """
    img_names_ = pd.read_csv(img_names_dir_)["img_name"]
    print(img_names_ is None)
    count = 0
    for index, value in img_names_.items():
        date_str, time_str = value.split("_")
        time = [int(num) for num in time_str.split("-")]
        if time[2] % 10 != 0:
            print(value, "first if")
            time[2] = round(time[2] / 10) * 10
            if time[2] == 60:
                time[2] = 50
                time = time_click(time, TICK_10SECONDS)
            rounded_time = "{:02d}-{:02d}-{:02d}".format(time[0], time[1], time[2])
            rounded_date_time = "{}_{}".format(date_str, rounded_time)
            file_name = "{}.jpg".format(value)
            file_new_name = "{}.jpg".format(rounded_date_time)
            os.rename(os.path.join(img_dir_, date_str, file_name), os.path.join(img_dir_, date_str, file_new_name))
            count += 1
            print(count)
    # extract_img_names(img_dir_, data_dir_)


def img_data_check(img_names_dir_):
    """
    Check how many images is missing, print the number of missing images out.
    :param img_names_dir_: path of the image names list csv
    """

    img_names_ = pd.read_csv(img_names_dir_)
    img_names_ = img_names_["img_name"]
    current_date = []
    current_time = []
    missing_count = 0
    missing_day = {}

    for index, value in img_names_.items():
        date_str, time_str = value.split("_")
        date = [int(num) for num in date_str.split("-")]
        time = [int(num) for num in time_str.split("-")]
        if current_date != date:
            current_date = date
            current_time = time
        else:
            current_time = time_click(current_time, TICK_10SECONDS)
            while current_time != time:

                # Refine the image names first, thus this part is unneeded

                # if time[2] % 10 != 0:
                #     print("oh!")
                #     time[2] = round(time[2] / 10) * 10
                #     if time[2] == 60:
                #         time[2] = 50
                #         time = time_click(time, TICK_10SECONDS)
                #     if current_time == time:
                #         break

                missing_count += 1
                if date_str not in missing_day:
                    missing_day[date_str] = 1
                else:
                    missing_day[date_str] += 1
                print(current_date, current_time)
                current_time = time_click(current_time, TICK_10SECONDS)

    print(missing_day)
    print("Number of images: {}".format(img_names_.size))
    print("Missing values: {}".format(missing_count))
    print("Missing percentage: {:.2%}".format(missing_count / img_names_.size))

    missing_percent = [missing_count, img_names_.size - missing_count]
    plt.pie(missing_percent, labels=["Missing images\n{}".format(missing_count),
                                     "Existed images\n{}".format(img_names_.size - missing_count)],
            colors=["#65a479", "#5d8ca8"], autopct="%.2f%%\n")
    plt.title("Percentage of missing images in dataset (2.5 months)")
    plt.show()


def csv_time_format_change(raw_data_dir_, data_folder_dir_, name="right_time_data.csv"):
    def time_format_change(date_time: str) -> str:
        date_time_obj = datetime.strptime(date_time, "%d/%m/%Y %I:%M:%S %p")
        return date_time_obj.strftime("%Y-%m-%d_%H-%M-%S")

    raw_data = pd.read_csv(raw_data_dir_)
    raw_data["DateTime"] = raw_data["DateTime"].apply(time_format_change)
    raw_data.to_csv(os.path.join(data_folder_dir_, name), index=False)


if __name__ == "__main__":
    # data_directory = "../pv_data"
    # img_directory = "../sky_image"
    # raw_data_dir = "../pv_data/raw_all_data.csv"
    # img_names_dir = "../index/img_names.csv"
    # formatted_data_dir = "../pv_data/right_time_data.csv"
    # other_data_dir = "../index"
    # concat_pv_csv(data_directory)
    # csv_time_format_change(raw_data_dir, data_directory)
    # select_data_with_img(img_directory, formatted_data_dir, data_directory)
    # extract_img_names(img_directory, other_data_dir)
    # round_img_name(img_names_dir, img_directory, data_directory)
    # select_data_without_img(img_names_dir, formatted_data_dir, data_directory)

    # from pv_data_process import ABSOLUTE_FILE_DIR, ABSOLUTE_IMG_DIR_1, ABSOLUTE_INDEX_DIR
    #
    # extract_img_names(ABSOLUTE_IMG_DIR_1, ABSOLUTE_INDEX_DIR, name="img_names_2020.csv")

    ABSOLUTE_INDEX_DIR = "/scratch/itee/uqsxu13/2020_data/2020_index/img_names.csv"
    ABSOLUTE_IMG_DIR_1 = "/scratch/itee/uqsxu13/2020_data/2020_gatton_1"

    extract_img_names(ABSOLUTE_IMG_DIR_1, "/scratch/itee/uqsxu13/2020_data/2020_index/")
    # round_img_name(ABSOLUTE_INDEX_DIR, ABSOLUTE_IMG_DIR_1, None)

    # select_data_with_img(img_names_dir, formatted_data_dir, data_directory)

import os

TICK_10SECONDS = 10
TICK_1SECOND = 1

IMAGE_FOLDER = os.path.join(os.pardir, "sky_image")


def time_click(time_, tick_):
    """
    past some seconds
    :param tick_: how many seconds to past, tick < 60
    :param time_: the last time list of integers
    :return: a new time list after 10s
    """
    hour, minute, sec = time_
    sec += tick_
    if sec == 60:
        sec = 00
        minute += 1
        if minute == 60:
            minute = 00
            hour = hour + 1 if hour != 23 else 00

    return [hour, minute, sec]


def get_path_from_datetime(datetime: str):
    date, time = datetime.split("_")
    path = os.path.join(IMAGE_FOLDER, date, datetime + ".jpg")
    return path

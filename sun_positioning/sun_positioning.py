import math
import os

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

GATTON_LATITUDE = -27.5596
GATTON_LONGITUDE = 152.3404
GMT10_MERIDIAN_LONGITUDE = 150
CROP_SIZE = (271, 0, 1822, 1516)


class SunPositionCalculator:

    def __init__(self, date_time: str):
        self.date, self.time = date_time.split("_")
        self.year, self.month, self.day = [int(i) for i in self.date.split("-")]
        self.hour, self.minute, self.second = [int(i) for i in self.time.split("-")]

    def is_leap_year(self):
        year = self.year
        if year % 4 == 0:
            if year % 100 == 0:
                if year % 400 == 0:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def nth_day_of_year(self):
        year = self.year
        month = self.month
        day = self.day

        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if self.is_leap_year():
            months[1] = 29

        days = 0

        for i in range(1, 13):
            if i == month:
                for j in range(i - 1):
                    days += months[j]
        days += day

        return days

    def declination_angle(self):
        nth_day = self.nth_day_of_year()
        delta = 23.45 * (self.sin(360 * (284 + nth_day) / 365))
        return delta

    def local_solar_time(self):
        nth_day = self.nth_day_of_year()
        # e: Equation of time
        b = 360 * (nth_day - 81) / 364
        e = 0.165 * self.sin(2 * b) - 0.126 * self.cos(b) - 0.025 * self.sin(b)
        clock_time = self.hour + self.minute / 60 + self.second / 3600
        # Lstd: Standard meridian of the local time zone [degrees east]
        lstd = GMT10_MERIDIAN_LONGITUDE
        # Lloc : Longitude of actual location [degrees east]
        lloc = GATTON_LONGITUDE
        # dt: daylight savings time
        dt = 0
        local_solar_time = clock_time - ((lstd - lloc) / 15) + e - dt

        return local_solar_time

    def time_angle(self):
        time_angle = 15 * (self.local_solar_time() - 12)
        return time_angle

    def solar_altitude_angle(self):
        solar_altitude_angle_rad = math.asin(self.sin(GATTON_LATITUDE)
                                             * self.sin(self.declination_angle())
                                             + self.cos(GATTON_LATITUDE)
                                             * self.cos(self.declination_angle())
                                             * self.cos(self.time_angle()))

        solar_altitude_angle = math.degrees(solar_altitude_angle_rad)

        return solar_altitude_angle

    def azimuth_angle(self):
        azimuth_angle_rad = math.acos((self.sin(self.declination_angle())
                                       * self.cos(GATTON_LATITUDE)
                                       - self.cos(self.declination_angle())
                                       * self.sin(GATTON_LATITUDE)
                                       * self.cos(self.time_angle()))
                                      / self.cos(self.solar_altitude_angle()))
        azimuth_angle = math.degrees(azimuth_angle_rad)
        if self.local_solar_time() > 12:
            azimuth_angle = -azimuth_angle
        return azimuth_angle

    @staticmethod
    def sin(degree):
        return math.sin(math.radians(degree))

    @staticmethod
    def cos(degree):
        return math.cos(math.radians(degree))


class SunPositionDraw(SunPositionCalculator):

    def __init__(self, image_dir: str):
        separator = os.sep
        if "/" in image_dir:
            separator = "/"
        date_time = image_dir.split(separator)[-1].split(".")[0]
        super(SunPositionDraw, self).__init__(date_time)
        self.image = Image.open(image_dir)
        self.image = self.image.crop(CROP_SIZE)
        self.draw = ImageDraw.Draw(self.image, "RGBA")
        self._width, self._height = self.image.size
        self._center_x = self._width / 2
        self._center_y = self._height / 2
        self._radius = self._center_y
        # self.draw.line((center_x, center_y, center_x, height))

        self._x_diff = self.sin(self.azimuth_angle()) * self._radius
        self._y_diff = self.cos(self.azimuth_angle()) * self._radius

        self._x_diff = self._x_diff * (1 - (self.solar_altitude_angle() / 90))
        self._y_diff = self._y_diff * (1 - (self.solar_altitude_angle() / 90))

        solar_angle_ratio = self.solar_altitude_angle() / 90
        distortion_x = solar_angle_ratio * 0.12 + 1
        # distortion_y = (math.pow(self.solar_altitude_angle() / 90, 4)) * 0.05 + 1

        self._x_diff *= distortion_x
        # self._y_diff /= distortion_y

        self._sun_center = (self._center_x + self._x_diff, self._center_y + self._y_diff)

    def get_sun(self) -> tuple:
        return self._sun_center

    def show_line(self):
        self.draw.line((self._center_x, self._center_y, self._sun_center[0], self._sun_center[1]), fill="red")
        self.image.show()

    def draw_circle(self, color=(40, 40, 60)):
        radius = 80
        sun_x = self._sun_center[0]
        sun_y = self._sun_center[1]

        left_up = (sun_x - radius, sun_y - radius)
        right_down = (sun_x + radius, sun_y + radius)
        points = [left_up, right_down]
        self.draw.ellipse(points, fill=color)
        # Cover the glare by shadow circles
        # self.draw.ellipse([(sun_x - 2*radius, sun_y - 2*radius), (sun_x + 2*radius, sun_y + 2*radius)],
        #                   fill=(0, 0, 10, 50))
        # self.draw.ellipse([(sun_x - 3*radius, sun_y - 3*radius), (sun_x + 3*radius, sun_y + 3*radius)],
        #                   fill=(0, 0, 10, 100))
        # self.image.show()
        return self.image


if __name__ == "__main__":
    heavy_cloud = "../sky_image/2018-07-02/2018-07-02_09-43-20.jpg"
    sink_cloud = "../sky_image/2018-07-11/2018-07-11_08-37-00.jpg"
    motion_test_1 = '../sky_image/2018-07-04/2018-07-04_12-08-40.jpg'
    motion_test_2 = '../sky_image/2018-07-04/2018-07-04_12-08-30.jpg'
    half_covered = '../sky_image/2018-07-04/2018-07-04_12-31-20.jpg'
    full_covered = '../sky_image/2018-07-04/2018-07-04_12-32-40.jpg'
    sun_rise = "../sky_image/2018-07-04/2018-07-04_06-50-00.jpg"
    edge_clear = "../sky_image/2018-07-04/2018-07-04_15-18-00.jpg"
    edge_cloudy = "../sky_image/2018-07-04/2018-07-04_08-51-20.jpg"
    clear = "../sky_image/2018-08-20/2018-08-20_10-00-00.jpg"

    # sun_position = SunPositionCalculator("2018-09-06_11-31-10")
    # print("nth day of the year: ", sun_position.nth_day_of_year())
    # print("Declination angle: ", sun_position.declination_angle())
    # print("Local solar time: ", sun_position.local_solar_time())
    # print("Time angle: ", sun_position.time_angle())
    # print("Solar altitude angle: ", sun_position.solar_altitude_angle())
    # print("Azimuth angle: ", sun_position.azimuth_angle())
    sun_positioning = SunPositionDraw(motion_test_1)
    # sun_positioning.show_line()
    plt.imshow(sun_positioning.draw_circle())
    plt.show()

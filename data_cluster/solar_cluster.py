import numpy.random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

numpy.random.seed(0)

SOLAR_DIR = "C:/Users/s4544852/Desktop/gatton PV data/exogenous/gatton_2020_solar_irradiance.csv"

solar = pd.read_csv(SOLAR_DIR)[22:127]
solar_mon = solar["Month"]
solar_mon = ["2020-{:02d}".format(x) for x in solar_mon]
solar_mon = np.array(solar_mon)
solar_day = solar["Day"]
solar_day = ["{:02d}".format(x) for x in solar_day]
solar_day = np.array(solar_day)
solar_num = solar["Daily global solar exposure (MJ/m*m)"]
solar_num = solar_num.to_numpy()
kmeans = KMeans(n_clusters=3)
a = kmeans.fit_predict(solar_num.reshape(-1, 1))
# print(a)
mon_sun = solar_mon[np.where(a == 0)]
day_sun = solar_day[np.where(a == 0)]
sun_list = [mon_sun, day_sun]
date_sun = tuple(np.apply_along_axis('-'.join, 0, sun_list).tolist())
mon_over = solar_mon[np.where(a == 1)]
day_over = solar_day[np.where(a == 1)]
over_list = [mon_over, day_over]
date_over = tuple(np.apply_along_axis('-'.join, 0, over_list).tolist())
mon_cld = solar_mon[np.where(a == 2)]
day_cld = solar_day[np.where(a == 2)]
cld_list = [mon_cld, day_cld]
date_cld = tuple(np.apply_along_axis('-'.join, 0, cld_list).tolist())
# print(len(date_sun))
# print(len(date_cld))
# print(len(date_over))

index_date = pd.read_csv("C:/Users/s4544852/Desktop/gatton PV data/index_2020/data_2020_interval.csv")["DateTime"]

index_sun = index_date[index_date.str.startswith(date_sun)]
index_cld = index_date[index_date.str.startswith(date_cld)]
index_over = index_date[index_date.str.startswith(date_over)]
index_sun.index.name = "old_idx"
index_cld.index.name = "old_idx"
index_over.index.name = "old_idx"

# print(index_sun)
# print(index_cld)
# print(index_over)

# index_sun.to_csv("C:/Users/s4544852/Desktop/gatton PV data/index_2020/date_sunny.csv", )
# index_cld.to_csv("C:/Users/s4544852/Desktop/gatton PV data/index_2020/date_cloudy.csv")
# index_over.to_csv("C:/Users/s4544852/Desktop/gatton PV data/index_2020/date_overcast.csv")

old_idx = pd.read_csv("C:/Users/s4544852/Desktop/gatton PV data/index_2020/date_sunny.csv")["old_idx"]
index_test = index_date[index_date.index.isin(old_idx)]
print(old_idx)
print(index_test)

# print(len(index_sun))
# print(len(index_cld))
# print(len(index_over))
exit()

# print(len(solar_num[a == 1]))
# print(len(solar_num[a == 2]))

print(solar_num[a == 0].mean())
print(solar_num[a == 1].mean())
print(solar_num[a == 2].mean())

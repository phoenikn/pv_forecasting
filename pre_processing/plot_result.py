import pandas as pd
import matplotlib.pyplot as plt


a = pd.read_csv("../pv_data/data_has_no_value.csv", header=None)
b = a["DateTime"].apply(lambda x: x.split("_")[1])
b = b.apply(lambda x: x.split("-")[0])
b = b.astype(int)
b = b[b >= 8]
b = b[b <= 15]
a = a.iloc[b.index_df]
a = a["DateTime"]
a = a.apply(lambda x: x.split("_")[0])
a = a.value_counts()
print(a)

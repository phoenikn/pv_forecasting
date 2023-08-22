import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load/split your data
pv_data = pd.read_csv("C:/Users/s4544852/Desktop/gatton PV data/index_2020/data_2020_interval.csv")
pv_data = pv_data["Power(kW)"].to_numpy()[10000:10010]

# Fit your model
model = pm.auto_arima(pv_data[:5], seasonal=False)

# make your forecasts
forecasts = model.predict(5)  # predict N steps into the future

# Visualize the forecasts (blue=train, green=forecasts)
x = np.arange(10)
plt.plot(x[:5], pv_data[:5], c='blue')
plt.plot(x[5:10], forecasts, c='green')
plt.plot(x[5:10], pv_data[5:], c='red')
plt.show()
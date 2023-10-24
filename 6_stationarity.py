import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

from make_stationary import make_stationary

df = pd.read_csv('data/airline-passengers.csv', index_col='Month', parse_dates=True)

# Using a function to find the best diff to obtain stationarity
diff = make_stationary(df['Passengers'])['differencing_order']
print("Diff:", diff)

# ADF stationarity test
# Returns: {Test statistic, P-value, Num lags used, {Critical values}, Estmation of maximized information criteria}
print("\033[1;32;40mOriginal Data:\033[0;37;40m",
      adfuller(df['Passengers'])) # P value is like .99 which is WAY more than .05

df['Passengers_Diff1'] = df['Passengers'].diff()
df['Passengers_Diff2'] = df['Passengers'].diff(2) # 2nd order difference
df = df.dropna()

adf_diff_1 = adfuller(df['Passengers_Diff1'])
adf_diff_2 = adfuller(df['Passengers_Diff2'])

print("\033[1;32;40mStationarized Data (Order 1):\033[0;37;40m",adf_diff_1) # P is .053
print("\033[1;32;40mStationarized Data (Order 2):\033[0;37;40m",adf_diff_2) # P is .038 (successfully stationarized)
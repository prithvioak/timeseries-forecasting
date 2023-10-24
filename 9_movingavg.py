import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

df = pd.read_csv('data/airline-passengers.csv', index_col='Month', parse_dates=True)

df['MA3'] = df['Passengers'].rolling(window=3,min_periods=1).mean()
df['MA6'] = df['Passengers'].rolling(window=6,min_periods=1).mean()
df['MA12'] = df['Passengers'].rolling(window=12,min_periods=1).mean()

plt.title('Airline passengers moving averages', size=20)
plt.plot(df['Passengers'], label='Original')
plt.plot(df['MA3'], color='gray', label='MA3')
plt.plot(df['MA6'], color='orange', label='MA6')
plt.plot(df['MA12'], color='red', label='MA12')
plt.legend()
plt.show()


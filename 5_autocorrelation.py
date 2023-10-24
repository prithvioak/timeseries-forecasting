import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

# Dataset
df = pd.read_csv('data/airline-passengers.csv', index_col='Month', parse_dates=True)

## BELOW: DECOMPOSITION
decomposed = seasonal_decompose(df, model='multiplicative')
decomposed.plot()
plt.show()

## For measuring autocorrelation, first make stationary, then use acl from statsmodels!
df['Passengers_Diff'] = df['Passengers'].diff(periods=1) # Stationary
df = df.dropna()
plt.title('Airline Passengers dataset', size=20)
plt.plot(df['Passengers'], label='Passengers')
plt.plot(df['Passengers_Diff'], label='Passengers_Diff')
plt.show()
# Autocorrelation
acf_values = acf(df['Passengers_Diff'])
plot_acf(df['Passengers_Diff'])
plt.show()
## High ac value at period 12 (because high autocorrelation from one year to next)

## Partial autocorrelation (pacf): Only the direct effect is shown, and all intermediary effects are removed.
plot_pacf(df['Passengers_Diff'])
plt.show()

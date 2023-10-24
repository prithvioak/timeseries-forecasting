import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

from make_stationary import make_stationary
from ar_order import ar_order

df = pd.read_csv('data/airline-passengers.csv', index_col='Month', parse_dates=True)
df.index= pd.to_datetime(df.index)

## Finding order of differencing needed
stationary_req = make_stationary(df['Passengers'])
# print("Order chosen:", stationary_req['differencing_order'], "whith p-value", stationary_req['p_val'])
d = stationary_req['differencing_order']
df['Passengers_Diff_n'] = df['Passengers'].diff(d) # Might not need to difference manually
print("d:",d)

## For this, can also use (find pmdarima thing):
# from pmdarima.arima.utils import ndiffs
# ## Adf Test
# ndiffs(y, test='adf')  # 2

# # KPSS test
# ndiffs(y, test='kpss')  # 0

# # PP test:
# ndiffs(y, test='pp')  # 2

## Finding the order of AR needed
test_size = 12

df_train = df[:-test_size]
df_test = df[-test_size:]

# import warnings
# warnings.filterwarnings('ignore')
# p, p_err = ar_order(df_train['Passengers'], df_test['Passengers'])
# warnings.filterwarnings('default')
plot_pacf(df['Passengers'].diff().dropna())
plt.show()
## Observing pacf plot, we notice that the first line is above the significance level
## The second one is as well, but not by much. So, p = 1.
p = 1
print("p:",p)

plot_acf(df['Passengers'].diff().dropna())
plt.show()
## Observing acf plot, we notice that the first two lines are above the significance level.
q = 2
print("q:",q)

## If your series is slightly under differenced, adding one or more additional AR terms usually makes it up.
## Likewise, if it is slightly over-differenced, try adding an additional MA term.

model = ARIMA(df_train['Passengers'], order=(p,d,q))
model_fit = model.fit()
predictions = model_fit.forecast(steps=12)
# df_test['forecast']=model_fit.predict(dynamic=True)
# predictions_df = pd.DataFrame(index=df_test.index, data=predictions[0])

## Plot
plt.title('Airline passengers MA(1) predictions', size=20)
plt.plot(df_train['Passengers'], label='Training data')
plt.plot(df_test['Passengers'], color='gray', label='Testing data')
plt.plot(predictions, color='orange', label='Predictions')
plt.legend(); plt.show()
from fbprophet import Prophet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

import logging
logger = logging.getLogger('fbprophet')
logger.setLevel(logging.DEBUG)

Prophet()
# print(m.stan_backend)

df = pd.read_csv('data/airline-passengers.csv', index_col='Month', parse_dates=True)
df.index= pd.to_datetime(df.index)

ts = pd.DataFrame({'ds':df.index,'y':df.Passengers}).dropna()
ts.index = pd.to_datetime(ts.index)
# print(ts.head(1))

test_size = 24

ts_train = ts[:-test_size]
ts_test = ts[-test_size:]

# prophet = Prophet()
# prophet.fit(ts_train)
# future = prophet.make_future_dataframe(periods=24)
# forecast = prophet.predict(future)

# fig = prophet.plot(forecast)
# plt.show()


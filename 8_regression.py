
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv('data/airline-passengers.csv', index_col='Month', parse_dates=True)

rmse = lambda act, pred: np.sqrt(mean_squared_error(act, pred))
# mape you just get from sklearn.metrics.mean_absolute_percentage_error

# Arbitrary data
actual_passengers = [300, 290, 320, 400, 500, 350]
predicted_passengers = [291, 288, 333, 412, 488, 344]

print(f'RMSE: {rmse(actual_passengers, predicted_passengers)}')
print(f'MAPE: {mean_absolute_percentage_error(actual_passengers, predicted_passengers)}')
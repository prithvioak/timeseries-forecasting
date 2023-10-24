import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

df = pd.read_csv('data/airline-passengers.csv', index_col='Month', parse_dates=True)

test_size = 24

df_train = df[:-test_size]
df_test = df[-test_size:]

plt.title('Airline passengers train and test sets', size=20)
plt.plot(df_train, label='Training set')
plt.plot(df_test, label='Test set', color='orange')
plt.legend()
plt.show()
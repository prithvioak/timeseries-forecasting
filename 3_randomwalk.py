import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from statsmodels.graphics.tsaplots import plot_acf
import random

random_walk = [0 for i in range(1000)]
for i in range(1,1000):
    random_walk[i] = random_walk[i-1] + random.choice([-1,1])

plt.title("Random walk plot")
plt.plot(random_walk)
plt.show()

# Plotting autocorrelation
plot_acf(np.array(random_walk))
plt.show()

# MAKE TIME SERIES STATIONARY BY TAKING DIFF
s_random_walk = pd.Series(random_walk)
s_random_walk_diff = s_random_walk.diff().dropna() # drop null values as well
print(s_random_walk_diff.head())

plt.title('Random Walk First Order Difference')
plt.plot(s_random_walk_diff)
plt.show()

# Plotting autocorrelation of difference to verify stationarity
plot_acf(np.array(s_random_walk_diff))
plt.show()
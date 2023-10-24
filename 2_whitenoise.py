import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.graphics.tsaplots import plot_acf
# from cycler import cycler

# Generate white noise
white_noise = np.random.randn(1000)
# plot
plt.title("White noise plot")
plt.plot(white_noise)
plt.show()

# plotting standard deviation and other stuff for 20 chunks of 50
plt.title("White noise mean and stdev")
plt.plot([white_noise.mean()] * 20, label = 'Global mean')
plt.scatter( x = np.arange(20), y = [np.mean(chunk) for chunk in np.split(white_noise, 20)], label='Mean')
plt.plot([white_noise.std()] * 20, label = 'Global stdev')
plt.scatter(x = np.arange(20), y = [np.std(chunk) for chunk in np.split(white_noise, 20)], label='Stdev')
plt.legend()
plt.show()

# Autocorrelation plots using statsmodels
plot_acf(np.array(white_noise))
plt.show()
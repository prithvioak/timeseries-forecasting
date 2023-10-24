import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

df = pd.read_csv('data/LTOTALNSA.csv', index_col='DATE', parse_dates=True)
print("\n\033[1;32;40mOriginal Data:\033[0;37;40m\n",df.head(),sep="")
yearly = df.resample(rule='Y').mean() # yearly aggregates of LTOTALNSA
print("\n\033[1;32;40mYearly aggregates:\033[0;37;40m\n",yearly.head(),sep="")
# yearly['Shift_3'] = yearly['LTOTALNSA'].shift(3) # Shifting the column down by 3
# yearly['Shift_neg4'] = yearly['LTOTALNSA'].shift(-4) # Shifting the column up by 4
# print(yearly.head())
# print(yearly.tail())

# Finding Moving Averages
df2 = df.copy()
df2['QuarterRolling'] = df2['LTOTALNSA'].rolling(window=3).mean()
df2['5-yearly'] = df2['LTOTALNSA'].rolling(window=60).mean()
print("\n\033[1;32;40mRolling averages added:\033[0;37;40m\n", df2.head(),sep="")

# Differencing (finding differences between consecutive or second order or third order etc)
# E.g if period = 1, difference between consec obs, 2 then diff between first and third etc
df2['Diff_1'] = df2['LTOTALNSA'].diff(1)
df2['Diff_2'] = df2['LTOTALNSA'].diff(2)
print("\n\033[1;32;40mDifferencing added:\033[0;37;40m\n", df2.head(),sep="")

# Plotting
plt.title('Light weight vehicle sales', size=20)
plt.xlabel('Time Period', size=14)
plt.ylabel('Number of units sold in 000', size=14)
plt.plot(df2[['LTOTALNSA','5-yearly']]['1990-01-01':'2005-01-01']) # From 1990 to 2005
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

## FUNCTION TO CHECK STATIONARITY
def make_stationary(data: pd.Series, alpha: float = 0.05, max_diff_order: int = 10) -> dict:
    # Test to see if the time series is already stationary
    if adfuller(data)[1] < alpha:
        return {
            'differencing_order': 0,
            'time_series': np.array(data),
            'p_val': (adfuller(data))[1]
        }
    
    p = 1
    diff = 0
    stationary_series = None
    # Test for differencing orders from 1 to max_diff_order (included)
    for i in range(1, max_diff_order + 1):
        # Perform ADF test
        diff_series = data.diff(i).dropna()
        result = adfuller(diff_series)
        # Append P-value
        if result[1] < alpha:
            diff = i
            p = result[1]
            stationary_series = np.array(diff_series)
            return {
                'differencing_order': diff,
                'time_series': stationary_series,
                'p_val': p
            }
    return
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

## THIS DOES NOT WORK WELL... Yet
def ar_order(data_train: pd.Series, data_test: pd.Series, max_order: int = 20):
    lowest_err = float('inf')
    corr_p = 0
    for p in range(1, max_order + 1):
        model = AutoReg(data_train, p).fit()
        preds = model.predict(
            start=len(data_train),
            end=len(data_train) + len(data_test) - 1,
            dynamic=False
        )
        error = mean_squared_error(data_test, preds, squared=False)
        if error < lowest_err:
            lowest_err = error
            corr_p = p
    
    return corr_p, lowest_err
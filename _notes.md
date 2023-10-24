
# IMPORTANT NOTES: Time Series Forecasting

## Time series decomposition
* Trend: general movement over time
* Seasonal: behaviors captured in individual __seasonal periods__
* Residual: everything not captured by trend and seasonal components (like error)
* Cyclical Component: Trends with no set repetition over a particular period of time. A cycle refers to the period of booms and slums of a time series. Cycles do not exhibit a seasonal variation but generally occur over a time period of 3 to 12 years depending on the nature of the time series.

__There are two techniques for combining time series components__:
* Additive: individual components (trend, seasonality, and residual) are added together.
    * Indicates a linear trend, and an additive seasonality indicates the same frequency (width) and amplitude (height) of seasonal cycles.
* Multiplicative: individual components are multiplied together.
    * Indicates a non-linear trend (curved trend line) as well as increasing/decreasing frequency (width) and/or amplitude (height) of seasonal cycles.

__Both trend and seasonality can be additive or multiplicative__:
* __Additive trend and additive seasonality__
Additive trend means the trend is linear (straight line), and additive seasonality means there aren’t any changes to widths or heights of seasonal periods over time.
![Alt text](https://miro.medium.com/v2/resize%3Afit%3A1400/format%3Awebp/1%2AHR0qLWq-u9JEJFJLaY5peg.jpeg)
* __Additive trend and multiplicative seasonality__
Additive trend means the trend is linear (straight line), and multiplicative seasonality means there are changes to widths or heights of seasonal periods over time.
![Alt text](https://miro.medium.com/v2/resize%3Afit%3A1400/format%3Awebp/1%2AMWBqxLG-DWlEIr8ZXuBg_g.jpeg)
* __Multiplicative trend and additive seasonality__
Multiplicative trend means the trend is not linear (curved line), and additive seasonality means there aren’t any changes to widths or heights of seasonal periods over time.
![Alt text](https://miro.medium.com/v2/resize%3Afit%3A1400/format%3Awebp/1%2A-v7RvOBvBA_vev0p5yG8hw.jpeg)
* __Multiplicative trend and multiplicative seasonality__
Multiplicative trend means the trend is not linear (curved line), and multiplicative seasonality means there are changes to widths or heights of seasonal periods over time.
![Alt text](https://miro.medium.com/v2/resize%3Afit%3A1400/format%3Awebp/1%2AdeKww-e91qLJUQX2T9rapQ.jpeg)

## Choosing the right model to use:
* If the ACF plot declines gradually and the PACF drops instantly, use Auto Regressive model.
* If the ACF plot drops instantly and the PACF declines gradually, use Moving Average model.
* If both ACF and PACF decline gradually, combine Auto Regressive and Moving Average models (ARMA).
* If both ACF and PACF drop instantly (no significant lags), it’s likely you won’t be able to model the time series.

```
pacf_values = pacf(df['Passengers_Diff'])
# take the diff, need stationary
acf_values = acf(df['Passengers_Diff'])
```

A _stationary process_ is a stochastic process whose unconditional __joint probability distribution does not change when shifted in time__. Consequently, parameters such as __mean and variance also do not change over time__.
A time series has to satisfy the following conditions to be considered stationary:
* __Constant mean__ — average value doesn’t change over time.
* __Constant variance__ — variance doesn’t change over time.
* __Constant covariance__ — covariance between periods of identical length doesn’t change over time.

<strong>Most forecasting algorithms assume a series is stationary.</strong>
You can test for stationarity using the <strong>Augmented Dickey-Fuller Test.</strong>

In Python, the ADF test returns the following:
* Test statistic
* P-value
* Number of lags used
* 1%, 5%, and 10% critical values
* Estimation of the maximized information criteria (idk)
If the returned __P-value is higher than 0.05, the time series isn’t stationary__. 0.05 is the standard threshold, but you’re free to change it.

## Evaluation Metrics
* AIC, or _Akaike Information Criterion_, shows you how good a model is relative to the other models. AIC penalizes complex models in favor of simple ones. (e.g same performance but less parameters chosen). Formula: $$AIC=2k-2\ln(\hat{L})$$ Where $k$ is the number of parameters in the model, $\hat{L}$ is the maximum value of the likelihood function for the model. (Lower value means better)

* BIC, or _Bayesian Information Criterion_, is an estimate of a function of the posterior probability of a model being true under a ceratin Bayesian setup. (Lower value means better) $$BIC = k\ln(n)-2\ln(\hat{L})$$

### General Regression Metrics
* RMSE — Root Mean Squared Error (RMSE will tell you how many units you can expect the model to miss in every forecast)
* MAPE — Mean Absolute Percentage Error (MAPE value of 0.02 means your forecasts are 98% accurate)

## Forecasting
### Moving Averages
![Alt text](https://miro.medium.com/v2/resize%3Afit%3A1168/format%3Awebp/1%2AswwgajGfXIEqSNdiRQKigg.png)
t is time period, s is size of sliding window

For t < s, the average of everything available is calculated (e.g the first value for MA is just the first observation)

`pandas` won’t copy the value by default, but will instead return `NaN`. You can specify `min_periods=1` inside the `rolling()` functions to avoid this behavior.

There are a couple of problems with simple moving averages:

* __Lag__ — moving average time series always lags from the original one. Look at the peaks to verify that claim.
* __Noise__ — too small sliding window size won’t remove all noise from the original data.
* __Averaging issue__ — averaged data will never capture the low and high points of the original series due to, well, averaging.
* __Weighting__ — identical weights are assigned to all data points. This can be an issue as frequently the most recent values have more impact on the future.

### Exponential Weighted Moving Averages
EWMA applies weights to the values of a time series. More weight is applied to more recent data points, making them more relevant for future forecasts.


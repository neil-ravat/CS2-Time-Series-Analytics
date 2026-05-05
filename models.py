# pyrefly: ignore-errors
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model


def get_adf_test_results(series):
    """
    Performs the Augmented Dickey-Fuller test.
    Returns the ADF statistic, p-value, and a boolean indicating
    stationarity (p < 0.05).
    """
    r = adfuller(series.dropna())
    adf_stat = round(r[0], 4)
    p_value = round(r[1], 4)
    is_stationary = bool(p_value < 0.05)
    return adf_stat, p_value, is_stationary


def get_arima_forecast(series, order=(2, 1, 2), steps=12):
    """
    Fits an ARIMA model and returns the forecast, confidence
    intervals, and AIC.
    """
    fitted = ARIMA(series, order=order).fit()
    fc = fitted.get_forecast(steps=steps)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int()
    return fitted, fc_mean, fc_ci


def get_var_forecast(data, maxlags=12, steps=12):
    """
    Fits a VAR model, selecting the best lag order up to maxlags.
    Returns the optimal lag, the model fit, and the forecasted
    values DataFrame.
    """
    mv = VAR(data)
    best_lag = mv.select_order(maxlags=maxlags).selected_orders["aic"]

    if best_lag == 0:
        best_lag = 1

    vf = mv.fit(best_lag)

    fc_raw = vf.forecast(data.values[-best_lag:], steps=steps)
    fc_df = pd.DataFrame(fc_raw, columns=data.columns)

    return best_lag, vf, fc_df


def get_granger_causality(data, maxlag):
    """
    Performs Granger Causality tests for a given pair of variables.
    Returns a DataFrame summarizing the lag, F-statistic, p-value,
    and significance.
    """
    gc = grangercausalitytests(data, maxlag=maxlag)
    rows = []
    for lag, res in gc.items():
        f_stat = round(res[0]["ssr_ftest"][0], 3)
        p_val = round(res[0]["ssr_ftest"][1], 4)
        signal = "Significant" if p_val < 0.05 else "None"
        rows.append([lag, f_stat, p_val, signal])

    return pd.DataFrame(rows, columns=["Lag", "F-stat", "p-value", "Signal"])


def get_garch_volatility(series, p=1, q=1):
    """
    Fits a GARCH(p,q) model and returns the conditional volatility
    and model parameters. Assumes series is typically stationary.
    """
    # Scale by 1000 for stability during optimization
    gf = arch_model(series / 1000, vol="Garch", p=p, q=q).fit(disp="off")
    vol = gf.conditional_volatility * 1000
    prm = gf.params
    return vol, prm


def get_cross_correlation(series_x, series_y, max_lags=12):
    """
    Compute cross-correlation between two time series at specified lags.
    """
    import numpy as np
    from statsmodels.tsa.stattools import ccf

    aligned = pd.concat([series_x, series_y], axis=1).dropna()
    sx = aligned.iloc[:, 0]
    sy = aligned.iloc[:, 1]

    ccf_vals = ccf(sx, sy, adjusted=False)
    ccf_forward = ccf_vals[: max_lags + 1]
    ccf_backward = ccf(sy, sx, adjusted=False)[: max_lags + 1]

    lags = np.arange(-max_lags, max_lags + 1)
    corrs = np.concatenate([ccf_backward[1:][::-1], ccf_forward])

    return pd.DataFrame({"Lag": lags, "Correlation": corrs})


def compute_forecast_metrics(actual, predicted):
    """
    Calculate MAE, RMSE, and MAPE for forecast evaluation.
    """
    import numpy as np

    aligned = pd.concat([actual, predicted], axis=1).dropna()
    y_true = aligned.iloc[:, 0]
    y_pred = aligned.iloc[:, 1]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    )

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def get_acf_pacf_values(series, nlags=20):
    """
    Compute autocorrelation and partial autocorrelation function values.
    """
    from statsmodels.tsa.stattools import acf, pacf

    series_clean = series.dropna()
    acf_vals, acf_confint = acf(series_clean, nlags=nlags, alpha=0.05)
    pacf_vals, pacf_confint = pacf(series_clean, nlags=nlags, alpha=0.05)

    return acf_vals, acf_confint, pacf_vals, pacf_confint


def get_seasonal_decomposition(series, period=12):
    """
    Perform seasonal decomposition (trend, seasonal, residual).
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    res = seasonal_decompose(series.dropna(), period=period)
    return res.trend, res.seasonal, res.resid

"""
Core statistical modeling engine for CS2 Time Series Analytics.

This module encapsulates all high-end statistical logic, including stationarity tests (ADF),
univariate (ARIMA) and multivariate (VAR) forecasting, volatility modeling (GARCH),
and diagnostic tools (ACF/PACF, Seasonal Decomposition).
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, ccf, grangercausalitytests, pacf
from typing import Tuple, Dict, Any, List, Optional

__all__ = [
    "get_adf_test_results",
    "get_arima_forecast",
    "get_var_forecast",
    "get_granger_causality",
    "get_garch_volatility",
    "get_cross_correlation",
    "compute_forecast_metrics",
    "get_acf_pacf_values",
    "get_seasonal_decomposition",
    "get_rolling_statistics",
    "get_anomalies",
    "get_growth_rates",
]


def get_adf_test_results(series: pd.Series) -> Tuple[float, float, bool]:
    """
    Performs the Augmented Dickey-Fuller test.
    Returns the ADF statistic, p-value, and a boolean indicating
    stationarity (p < 0.05).
    """
    r = adfuller(series.dropna())
    adf_stat = round(float(r[0]), 4)
    p_value = round(float(r[1]), 4)
    is_stationary = bool(p_value < 0.05)
    return adf_stat, p_value, is_stationary


def get_arima_forecast(
    series: pd.Series,
    order: Tuple[int, int, int] = (2, 1, 2),
    steps: int = 12,
    alpha: float = 0.05,
) -> Tuple[Any, pd.Series, pd.DataFrame]:
    """
    Fits an ARIMA model and returns the forecast, confidence
    intervals, and AIC.
    """
    fitted = ARIMA(series, order=order).fit()
    fc = fitted.get_forecast(steps=steps)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=alpha)
    return fitted, fc_mean, fc_ci


def get_var_forecast(
    data: pd.DataFrame, maxlags: int = 12, steps: int = 12
) -> Tuple[int, Any, pd.DataFrame]:
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

    return int(best_lag), vf, fc_df


def get_granger_causality(data: pd.DataFrame, maxlag: int) -> pd.DataFrame:
    """
    Performs Granger Causality tests for a given pair of variables.
    Returns a DataFrame summarizing the lag, F-statistic, p-value,
    and significance.
    """
    gc = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    rows = []
    for lag, res in gc.items():
        f_stat = round(float(res[0]["ssr_ftest"][0]), 3)
        p_val = round(float(res[0]["ssr_ftest"][1]), 4)
        signal = "Significant" if p_val < 0.05 else "None"
        rows.append([lag, f_stat, p_val, signal])

    return pd.DataFrame(rows, columns=["Lag", "F-stat", "p-value", "Signal"])


def get_garch_volatility(
    series: pd.Series, p: int = 1, q: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """
    Fits a GARCH(p,q) model and returns the conditional volatility
    and model parameters. Assumes series is typically stationary.
    """
    # Scale by 1000 for stability during optimization
    gf = arch_model(series / 1000, vol="Garch", p=p, q=q).fit(disp="off")
    vol = gf.conditional_volatility * 1000
    prm = gf.params
    return vol, prm


def get_cross_correlation(
    series_x: pd.Series, series_y: pd.Series, max_lags: int = 12
) -> pd.DataFrame:
    """
    Compute cross-correlation between two time series at specified lags.
    """
    aligned = pd.concat([series_x, series_y], axis=1).dropna()
    sx = aligned.iloc[:, 0]
    sy = aligned.iloc[:, 1]

    ccf_vals = ccf(sx, sy, adjusted=False)
    ccf_forward = ccf_vals[: max_lags + 1]
    ccf_backward = ccf(sy, sx, adjusted=False)[: max_lags + 1]

    lags = np.arange(-max_lags, max_lags + 1)
    corrs = np.concatenate([ccf_backward[1:][::-1], ccf_forward])

    return pd.DataFrame({"Lag": lags, "Correlation": corrs})


def compute_forecast_metrics(
    actual: pd.Series, predicted: pd.Series
) -> Dict[str, float]:
    """
    Calculate MAE, RMSE, and MAPE for forecast evaluation.
    """
    aligned = pd.concat([actual, predicted], axis=1).dropna()
    y_true = aligned.iloc[:, 0]
    y_pred = aligned.iloc[:, 1]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    )

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def get_acf_pacf_values(
    series: pd.Series, nlags: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute autocorrelation and partial autocorrelation function values.
    """
    series_clean = series.dropna()
    acf_vals, acf_confint = acf(series_clean, nlags=nlags, alpha=0.05)
    pacf_vals, pacf_confint = pacf(series_clean, nlags=nlags, alpha=0.05)

    return acf_vals, acf_confint, pacf_vals, pacf_confint


def get_seasonal_decomposition(
    series: pd.Series, period: int = 12
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Perform seasonal decomposition (trend, seasonal, residual).
    """
    res = seasonal_decompose(series.dropna(), period=period)
    return res.trend, res.seasonal, res.resid


def get_rolling_statistics(
    series: pd.Series, window: int = 12
) -> Tuple[pd.Series, pd.Series]:
    """Calculate rolling mean and standard deviation."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return rolling_mean, rolling_std


def get_anomalies(series: pd.Series, threshold_z: float = 2.5) -> pd.Series:
    """Detect anomalies using Z-score."""
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    anomalies = series[np.abs(z_scores) > threshold_z]
    return anomalies


def get_growth_rates(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Calculate Month-over-Month and Year-over-Year growth rates."""
    mom = series.pct_change(1) * 100
    yoy = series.pct_change(12) * 100
    return mom, yoy

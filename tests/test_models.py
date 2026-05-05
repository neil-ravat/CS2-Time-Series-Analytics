import pandas as pd
import numpy as np
import pytest
from models import (
    get_adf_test_results,
    get_arima_forecast,
    get_var_forecast,
    get_granger_causality,
    get_garch_volatility,
)


def test_get_adf_test_results(sample_series):
    # A random walk is typically non-stationary
    stat, p_val, is_stationary = get_adf_test_results(sample_series)
    assert isinstance(stat, float)
    assert isinstance(p_val, float)
    assert isinstance(is_stationary, bool)

    # First difference should be stationary (white noise)
    diff_series = sample_series.diff().dropna()
    stat_diff, p_val_diff, is_stat_diff = get_adf_test_results(diff_series)
    assert is_stat_diff


def test_get_arima_forecast(sample_series):
    fitted, fc_mean, fc_ci = get_arima_forecast(sample_series, order=(1, 1, 1), steps=5)
    assert len(fc_mean) == 5
    assert fc_ci.shape == (5, 2)
    assert hasattr(fitted, "aic")


def test_get_var_forecast(sample_multivariate_data):
    # Use differenced data to ensure stationarity for VAR
    diff_data = sample_multivariate_data.diff().dropna()
    best_lag, vf, fc_df = get_var_forecast(diff_data, maxlags=3, steps=4)

    assert isinstance(best_lag, int)
    assert best_lag >= 0
    assert fc_df.shape == (4, 3)
    assert list(fc_df.columns) == ["Players", "Trends", "Twitch"]


def test_get_granger_causality(sample_multivariate_data):
    diff_data = sample_multivariate_data.diff().dropna()
    df_gc = get_granger_causality(diff_data[["Players", "Trends"]], maxlag=2)

    assert isinstance(df_gc, pd.DataFrame)
    assert not df_gc.empty
    assert list(df_gc.columns) == ["Lag", "F-stat", "p-value", "Signal"]


def test_get_garch_volatility(sample_series):
    diff_series = sample_series.diff().dropna()
    vol, prm = get_garch_volatility(diff_series, p=1, q=1)

    assert len(vol) == len(diff_series)
    assert "alpha[1]" in prm
    assert "beta[1]" in prm

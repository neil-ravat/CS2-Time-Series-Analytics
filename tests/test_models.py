import pytest
import pandas as pd
import numpy as np
from models import get_arima_forecast, get_var_forecast, get_adf_test_results


def test_check_stationarity(sample_series):
    adf_stat, p_value, is_stationary = get_adf_test_results(sample_series)
    assert isinstance(adf_stat, (float, np.floating))
    assert isinstance(p_value, (float, np.floating))
    assert isinstance(is_stationary, bool)


def test_arima_forecast(sample_series):
    # Test valid forecast
    fitted, fc_mean, fc_ci = get_arima_forecast(sample_series, steps=5)
    assert len(fc_mean) == 5
    assert len(fc_ci) == 5
    assert hasattr(fitted, "aic")


def test_var_forecast(sample_multivariate_data):
    # Test valid VAR forecast
    best_lag, vf, fc_df = get_var_forecast(sample_multivariate_data, steps=3)
    assert len(fc_df) == 3
    assert "Players" in fc_df.columns
    assert isinstance(best_lag, (int, np.integer))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_series():
    """Generate a sample random walk time series for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="MS")
    values = np.random.randn(100).cumsum() * 10000
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_multivariate_data():
    """Generate sample multivariate data for testing VAR/Granger."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="MS")
    df = pd.DataFrame(
        {
            "Players": np.random.randn(100).cumsum(),
            "Trends": np.random.randn(100).cumsum() + 10,
            "Twitch": np.random.randn(100).cumsum() - 5,
        },
        index=dates,
    )
    return df

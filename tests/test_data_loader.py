import pytest
import pandas as pd
import os
from data_loader import load_data, get_data_splits


def test_load_data_error():
    # Test behavior when file doesn't exist
    with pytest.raises(FileNotFoundError):
        load_data("non_existent.csv")


def test_data_processing(tmp_path):
    # Create a temporary CSV to test loading
    d = tmp_path / "Players - Sheet1.csv"
    d.write_text(
        "Month,Avg. Players,Date,Trend,Twitch Viewers\n"
        "July 2012,933,07-01-2012,93,\n"
        'August 2012,15475,08-01-2012,100,"25,000"\n'
    )

    df = load_data(str(d))

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "Date"
    assert df["Players"].iloc[0] == 933
    # Check that Twitch commas are handled
    assert df["Twitch"].iloc[1] == 25000

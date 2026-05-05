import pandas as pd
from data_loader import load_data, get_data_splits

def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ['Players', 'Trends', 'Twitch']
    assert isinstance(df.index, pd.DatetimeIndex)

def test_get_data_splits():
    df, df_full, df_multi = get_data_splits()
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_full, pd.DataFrame)
    assert isinstance(df_multi, pd.DataFrame)
    assert 'Twitch' not in df_full.columns
    assert 'Twitch' in df_multi.columns
    # Check that df_multi has no NaNs in Twitch
    assert not df_multi['Twitch'].isna().any()

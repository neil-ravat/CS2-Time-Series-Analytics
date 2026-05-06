# pyrefly: ignore-errors
import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_data(file_path=None):
    """
    Load and preprocess the CS2 Time Series dataset.
    Caches the result using Streamlit to avoid redundant disk I/O.
    """
    if file_path is None:
        base_path = Path(__file__).parent
        file_path = base_path / "Players - Sheet1.csv"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = ["Month", "Players", "Date", "Trends", "Twitch"]

    # Preprocess 'Twitch' column
    twitch_str = df["Twitch"].astype(str).str.replace(",", "").str.strip()
    df["Twitch"] = pd.to_numeric(twitch_str, errors="coerce")

    # Convert 'Date' to datetime and set as index
    df["Date"] = pd.to_datetime(df["Date"], format="%m-%d-%Y")
    df.set_index("Date", inplace=True)

    return df[["Players", "Trends", "Twitch"]].sort_index()


def get_data_splits():
    """
    Returns the primary dataframe along with the full overlap
    and multivariate overlap subsets.
    """
    df = load_data()
    df_full = df[["Players", "Trends"]].copy()
    df_multi = df.dropna(subset=["Twitch"]).copy()

    return df, df_full, df_multi

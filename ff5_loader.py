# ff5_loader.py
from pathlib import Path
import requests, zipfile, io, pandas as pd
from ff5_urls import FF5_CSV_URL        # ← dict you made in Step 1
import streamlit as st

@st.cache_data(show_spinner="Downloading factors…")
def load_ff5(region: str) -> pd.DataFrame:
    """
    Return a tidy DataFrame with columns:
    ['Date', 'Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    Dates converted to pandas Period ('M').
    """
    url = FF5_CSV_URL[region]          # raises KeyError if region unknown
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Un-zip in memory
    zf  = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
    df_raw = pd.read_csv(
        zf.open(csv_name),
        skiprows=3,            # first 3 lines are header junk
    )

    # Tidy
    df_raw.rename(columns=lambda c: c.strip().replace(" ", "_"), inplace=True)
    df_raw = df_raw.query("Date != 'Annual'")      # drop footer
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], format="%Y%m").dt.to_period("M")
    df_raw = df_raw.apply(pd.to_numeric, errors="coerce")
    return df_raw

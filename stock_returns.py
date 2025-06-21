# stock_returns.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os

# ─── Optional Refinitiv import & key setup ───────────────────────────────────
try:
    import refinitiv.dataplatform.eikon as ek
    HAS_REFINTIV = True
except ImportError:
    HAS_REFINTIV = False

if HAS_REFINTIV:
    try:
        ek.set_app_key(st.secrets["REFINITIV_APP_KEY"])
    except KeyError:
        st.warning(
            "REFINITIV_APP_KEY not found in Streamlit Secrets; live Refinitiv calls disabled."
        )
else:
    st.warning(
        "Refinitiv SDK not installed; live fetch of market data disabled."
    )

# ─── Fetch Daily Returns ──────────────────────────────────────────────────────
def fetch_daily_returns(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Fetch historical daily returns for a given ticker via Refinitiv Eikon.
    Requires REFINTIV_APP_KEY secret and refinitiv SDK installed.
    """
    if not HAS_REFINTIV:
        raise RuntimeError("Refinitiv API client not available")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    df = ek.get_timeseries(
        rics=ticker,
        fields="CLOSE",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="daily"
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df["Return"] = df["CLOSE"].pct_change()
    df.dropna(subset=["Return"], inplace=True)
    return df

# ─── Save to CSV ──────────────────────────────────────────────────────────────
def save_returns_to_csv(ticker: str, df: pd.DataFrame) -> Path:
    Path("attached_assets").mkdir(exist_ok=True)
    filename = f"returns_{ticker.replace('.', '_')}.csv"
    path = Path("attached_assets") / filename
    df.to_csv(path)
    st.info(f"Saved returns CSV: {path}")
    return path

# ─── Main Function to Loop Through Tickers ────────────────────────────────────
def main():
    tickers = ["CSL.AX", "FLT.AX", "SEK.AX", "WTC.AX", "XRO.AX","BOSSn.DE",
               "DHLn.DE","HFGG.DE","KGX.DE","SHLG.DE","TMV.DE","AIR.NZ","FBU.NZ","FCG.NZ",
                "MEL.NZ", "ADSK.O","DG","HSY","INTU.O","PYPL.O","URI" ]  # Add any more tickers
    for ticker in tickers:
        try:
            df = fetch_daily_returns(ticker)
            save_returns_to_csv(ticker, df)
        except Exception as e:
            print(f"❌ Failed for {ticker}: {e}")

# ─── Run If Standalone ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

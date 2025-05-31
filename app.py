#app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ─── 1) Your existing loader/grabber ────────────────────────────────────────────
YEAR_ROW = 10
COLS     = list(range(1,16))

def load_sheet(xlsx: Path, sheet: str):
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet, header=None, engine="openpyxl")
    except:
        return None, None
    if df.shape[0] <= YEAR_ROW or df.shape[1] <= max(COLS):
        return None, None
    years = df.iloc[YEAR_ROW, COLS].astype(int).tolist()
    return df, years

def grab_series(xlsx: Path, sheet: str, regex: str):
    df, years = load_sheet(xlsx, sheet)
    if df is None:
        return None
    col0 = df.iloc[:,0].astype(str).str.lower()
    mask = col0.str.contains(regex, regex=True, na=False)
    if not mask.any():
        return None
    row = df.loc[mask, :].iloc[0]
    return pd.to_numeric(row.iloc[COLS], errors="coerce").tolist()

@st.cache_data
def build_dataset():
    base = Path(__file__).parent
    rows = []
    #for country_dir in base.iterdir():
        #if not country_dir.is_dir():
            #continue
        #for xlsx in country_dir.glob("*.xlsx"):
    for xlsx in base.rglob("*.xlsx"):
       


            ticker = xlsx.stem
            _, years = load_sheet(xlsx, "Income Statement")
            if years is None:
                continue
            ebitda = grab_series(xlsx, "Income Statement", r"earnings before.*ebitda")
            capex  = grab_series(xlsx, "Cash Flow",         r"capital expenditure|capex")
            debt   = grab_series(xlsx, "Balance Sheet",     r"total debt|debt\b")
            cash   = grab_series(xlsx, "Balance Sheet",     r"cash and cash equivalents|cash$")
            ev     = grab_series(xlsx, "Financial Summary", r"^enterprise value\s*$")
            if None in (ebitda, capex, debt, cash, ev):
                continue

            for y,e,c,d,ca,v in zip(years, ebitda, capex, debt, cash, ev):
                rows.append({
                    "Ticker": ticker, "Year": y,
                    "EBITDA": e,       "CapEx":  c,
                    "Debt":   d,       "Cash":   ca,
                    "EV":     v,
                    "FCF":    e - c
                })
    df = pd.DataFrame(rows)
    df["EV/EBITDA"] = df["EV"] / df["EBITDA"].replace(0, pd.NA)
    return df

# ─── 2) Page setup & CSS ───────────────────────────────────────────────────────
st.set_page_config(page_title="🚀 Starship Finance Simulator", layout="wide")
st.markdown("""
  <style>
    body { background:#0d1117; color:#39ff14; font-family:'Courier New', monospace; }
    .stSlider>div>div>label { color:#0ff; }
    .stMetric { background: #010409; border:1px solid #30363d; border-radius:6px; }
  </style>
""", unsafe_allow_html=True)

df = build_dataset()
if df.empty:
    st.error("❌ No data found. Check your folders/sheets.")
    st.stop()



# ─── 3) Sidebar: selectors & sliders ────────────────────────────────────────────
tickers     = sorted(df["Ticker"].unique())
sel_tickers = st.sidebar.multiselect(
    "🔍 Companies",
    options=tickers,
    default=[]             # ← no pre‐selection
)

if not sel_tickers:
    st.sidebar.info("❗ Please select at least one company to continue.")
    st.stop()

# Clamp Year to only the years present for the selected tickers
years_avail  = df[df.Ticker.isin(sel_tickers)]["Year"].dropna().unique()
years_sorted = sorted(int(y) for y in years_avail)
if not years_sorted:
    st.sidebar.error("No years available for the selected companies.")
    st.stop()

sel_year = st.sidebar.slider(
    "📅 Year",
    min_value=years_sorted[0],
    max_value=years_sorted[-1],
    value=years_sorted[-1],
)

# ─── define your simulation sliders here ───────────────────────────────────────
st.sidebar.markdown("### 🎛 Simulations")
ebt_adj  = st.sidebar.slider("EBITDA Δ%",   -50, 50, 0)
cpx_adj  = st.sidebar.slider("CapEx Δ%",    -50, 50, 0)
debt_adj = st.sidebar.slider("Debt Δ%",     -50, 50, 0)
cash_adj = st.sidebar.slider("Cash Δ%",     -50, 50, 0)
ev_mult  = st.sidebar.slider("EV/EBITDA",    0.1, 50.0, 12.0, step=0.1)

# ─── 4) Filter & simulate ───────────────────────────────────────────────────────
base = df.query("Year==@sel_year & Ticker in @sel_tickers").copy()
if base.empty:
    st.warning("No data for that selection.")
    st.stop()

sim = base.copy()
for col, pct in [
    ("EBITDA", ebt_adj),
    ("CapEx",  cpx_adj),
    ("Debt",   debt_adj),
    ("Cash",   cash_adj),
]:
    sim[col] = sim[col] * (1 + pct/100)

sim["FCF"]       = sim["EBITDA"] - sim["CapEx"]
sim["EV"]        = sim["EBITDA"] * ev_mult
sim["EV/EBITDA"] = ev_mult


# ─── 4) Filter & simulate ───────────────────────────────────────────────────────
base = df.query("Year==@sel_year & Ticker in @sel_tickers").copy()
if base.empty:
    st.warning("No data for that selection.")
    st.stop()

sim = base.copy()
for col,p in [("EBITDA", ebt_adj), ("CapEx", cpx_adj),
              ("Debt", debt_adj), ("Cash", cash_adj)]:
    sim[col] = sim[col] * (1 + p/100)
sim["FCF"]       = sim["EBITDA"] - sim["CapEx"]
sim["EV"]        = sim["EBITDA"] * ev_mult
sim["EV/EBITDA"] = ev_mult

# ─── 5) Top metrics: one panel per ticker ──────────────────────────────────────

hist_metrics = [
    ("EBITDA",    "EBITDA",    "$ {:,.0f}"),
    ("CapEx",     "CapEx",     "$ {:,.0f}"),
    ("FCF",       "FCF",       "$ {:,.0f}"),
    ("EV",        "EV",        "$ {:,.0f}"),
    ("EV/EBITDA", "EV/EBITDA", "{:.2f}x"),
]

for ticker in sel_tickers:
    t_base = base[base.Ticker == ticker]
    if t_base.empty:
        # no data for this ticker in that year → skip entirely
        continue

    t_sim  = sim[sim.Ticker == ticker]

    st.markdown(f"## {ticker} – Year {sel_year}")

    # Historical
    st.markdown("### Historical Metrics")
    cols = st.columns(5)
    for (label, field, fmt), col in zip(hist_metrics, cols):
        if field == "EV/EBITDA":
            e_sum = t_base["EBITDA"].sum(skipna=True)
            val   = (t_base["EV"].sum(skipna=True) / e_sum) if e_sum else pd.NA
        else:
            val = t_base[field].sum(skipna=True)
        col.metric(label, fmt.format(val) if pd.notna(val) else "n/a")

    # Simulated
    st.markdown("### Simulated Metrics")
    cols = st.columns(5)
    for (label, field, fmt), col in zip(hist_metrics, cols):
        if field == "EV/EBITDA":
            e_sum    = t_base["EBITDA"].sum(skipna=True)
            hist_val = (t_base["EV"].sum(skipna=True) / e_sum) if e_sum else pd.NA
            sim_val  = t_sim["EV/EBITDA"].iloc[0] if (not t_sim.empty) else pd.NA
        else:
            hist_val = t_base[field].sum(skipna=True)
            sim_val  = t_sim[field].sum(skipna=True)

        if pd.isna(sim_val) or pd.isna(hist_val) or hist_val == 0:
            delta_str = ""
        else:
            delta     = sim_val / hist_val - 1
            delta_str = f"{delta:+.1%}"

        col.metric(
            label,
            fmt.format(sim_val) if pd.notna(sim_val) else "n/a",
            delta_str,
            help="FCF = EBITDA − CapEx" if field == "FCF" else "",
        )


# ─── 6) 3D Scatter (Ticker & 4-dec hover) ─────────────────────────────────────
st.markdown("### 🔭 3D Simulation: EBITDA vs CapEx vs EV")

plot_df = pd.concat([ base.assign(Type="Base"), sim.assign(Type="Simulated") ])
# plot_df["FCF_mag"]   = plot_df["FCF"].abs()
# plot_df["FCF_label"] = plot_df["FCF"].apply(lambda x: "Positive" if x >= 0 else "Negative")
# Compute magnitude, and replace any NaN with 0
plot_df["FCF_mag"]   = plot_df["FCF"].abs().fillna(0)

# Human‐readable label
plot_df["FCF_label"] = plot_df["FCF"].apply(lambda x: "Positive" if x >= 0 else "Negative")

fig3d = px.scatter_3d(
    plot_df,
    x="EBITDA", y="CapEx", z="EV",
    color="FCF_label",
    color_discrete_map={"Negative":"red","Positive":"green"},
    symbol="Type",
    size="FCF_mag", size_max=20,
    hover_name="Ticker",
    hover_data={
        "Type":      True,
        "EBITDA":    ":.4f",
        "CapEx":     ":.4f",
        "EV":        ":.4f",
        "Debt":      ":.4f",
        "Cash":      ":.4f",
        "EV/EBITDA": ":.4f",
        "FCF":       ":.4f",
        "FCF_mag":   False,
        "FCF_label": False,
    },
    template="plotly_dark",
    title=f"Year {sel_year}: Base vs Simulated"
)
fig3d.update_layout(
    height=600,
    margin=dict(l=0,r=0,t=40,b=0),
    scene=dict(xaxis_title="EBITDA",yaxis_title="CapEx",zaxis_title="EV")
)
st.plotly_chart(fig3d, use_container_width=True)

# ─── 7) EV/EBITDA & FCF Over Time ───────────────────────────────────────────────
st.markdown("### 🔄 EV/EBITDA & FCF Over Time")
time_df = df[df["Ticker"].isin(sel_tickers)].copy()
time_df["FCF"] = time_df["EBITDA"] - time_df["CapEx"]
fig2 = px.line(
    time_df,
    x="Year",
    y=["FCF","EV/EBITDA"],
    color="Ticker",
    markers=True,
    template="plotly_dark",
    labels={
      "value":"FCF (USD) / EV/EBITDA (x)",
      "variable":"Metric",
      "Ticker":"Company"
    },
)
st.plotly_chart(fig2, use_container_width=True)

# ─── 8) DataTable ───────────────────────────────────────────────────────────────
st.markdown("### 📊 Data Table")
st.dataframe(
    sim[["Ticker","Year","EBITDA","CapEx","FCF","EV","EV/EBITDA","Debt","Cash"]],
    use_container_width=True,
    height=300
)


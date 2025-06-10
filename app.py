# app.py
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

    for xlsx in base.rglob("*.xlsx"):
        ticker = xlsx.stem

        # get years
        _, years = load_sheet(xlsx, "Income Statement")
        if years is None:
            continue

        # grab core series
        ebitda    = grab_series(xlsx, "Income Statement", r"earnings before.*ebitda")
        capex     = grab_series(xlsx, "Cash Flow",         r"capital expenditure|capex")
        debt      = grab_series(xlsx, "Balance Sheet",     r"total debt|debt\b")
        cash      = grab_series(xlsx, "Balance Sheet",     r"cash and cash equivalents|cash$")
        ev        = grab_series(xlsx, "Financial Summary", r"^enterprise value\s*$")
        taxes_cf  = grab_series(xlsx, "Cash Flow",         r"income taxes\s*-\s*paid")

        # skip if any core series missing
        if None in (ebitda, capex, debt, cash, ev, taxes_cf):
            continue

        # compute ΔNWC from Balance Sheet
        curr_assets = grab_series(xlsx, "Balance Sheet", r"total current assets")
        curr_liab   = grab_series(xlsx, "Balance Sheet", r"total current liabilities")

        if curr_assets and curr_liab:
            nwc = [a - l for a, l in zip(curr_assets, curr_liab)]
            change_in_nwc = [0] + [nwc[i] - nwc[i-1] for i in range(1, len(nwc))]
        else:
            change_in_nwc = [0] * len(years)

        # pull interest (IS first, then CF)
        ie_is = grab_series(xlsx, "Income Statement", r"interest expense|finance costs")
        ie_cf = grab_series(xlsx, "Cash Flow",        r"interest\s*paid")
        interest_expense = ie_is if ie_is is not None else (ie_cf or [0] * len(years))

        # assemble rows
        for y, e, c, d, ca, v, t, nwc0, ie in zip(
            years, ebitda, capex, debt, cash, ev, taxes_cf,
            change_in_nwc, interest_expense
        ):
            rows.append({
                "Ticker":          ticker,
                "Year":            y,
                "EBITDA":          e,
                "CapEx":           c,
                "Debt":            d,
                "Cash":            ca,
                "EV":              v,
                "CashTaxesPaid":   t,
                "ChangeNWC":       nwc0,
                "InterestExpense": ie,
            })

    # build DataFrame
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    
    
        # historical ΔDebt & ΔCash
    df["ΔDebt"] = df.groupby("Ticker")["Debt"].diff().fillna(0)
    df["ΔCash"] = df.groupby("Ticker")["Cash"].diff().fillna(0)

    # Unlevered Free Cash Flow (FCFF)
    df["FCFF"] = df["EBITDA"] - df["CashTaxesPaid"] - df["ChangeNWC"] - df["CapEx"]

    # Legacy Free Cash Flow (FCF)
    #df["FCF"] = df["EBITDA"] - df["CashTaxesPaid"] - df["CapEx"]

    # free cash flow including working-capital changes
    df["FCF"] = (
    df["EBITDA"]
  - df["CashTaxesPaid"]
  - df["ChangeNWC"]
  - df["CapEx"]
)


    # Levered Free Cash Flow (FCFE)
    tax_rate = 0.21
    df["FCFE"] = df["FCFF"] - df["InterestExpense"] * (1 - tax_rate) + df["ΔDebt"] - df["ΔCash"]

    # EV/EBITDA
    df["EV/EBITDA"] = df["EV"] / df["EBITDA"].replace(0, pd.NA)

    return df


# ─── 2) Page setup & CSS ───────────────────────────────────────────────────────
st.set_page_config(page_title="🚀 Starship Finance Simulator", layout="wide")
st.markdown("""
<style>
  /* 1) Import a sci-fi font */
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');

  /* 2) Background & base text */
  body {
    background: radial-gradient(circle at center, #000011, #000);
    color: #39ff14;
    font-family: 'Orbitron', monospace;
  }

  /* 3) Make your metric cards wider and neon-glow */
  .stMetric {
  flex: 0 0 130px !important;
  min-width: 130px !important;
  max-width: 130px !important;

  /* make every card the same height */
  min-height: 113px !important;

  /* your existing styles */
  background: rgba(0, 0, 30, 0.8) !important;
  border: 2px solid #0ff !important;
  border-radius: 12px !important;
  padding: 8px !important;
  box-shadow: 0 0 8px #0ff, 0 0 16px #39ff14 !important;

  /* center value+delta vertically */
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
}


  /* Neon glow on the numbers */
  .stMetric .value {
    font-size: 1.6rem !important;
    text-shadow: 0 0 4px #39ff14 !important;
  }
  .stMetric .delta {
    font-size: 1.1rem !important;
    text-shadow: 0 0 4px #0ff !important;
  }

  /* 4) Force each metric-row into a no-wrap, scrollable flex strip */
  /* ─── Force each metric card to the same size ───────────────────────────────── */
.stMetric {
  flex: 0 0 220px !important;     /* exact width for every card */
  min-width: 220px !important;
  max-width: 220px !important;
  margin: 0 !important;           /* reset any default margins */
}

/* ─── Evenly distribute the cards in each row ───────────────────────────────── */
section[data-testid="metric-container"] > div {
  display: flex !important;
  flex-wrap: nowrap !important;
  justify-content: space-evenly !important;
  align-items: stretch !important;
  gap: 16px !important;           /* space between cards */
  overflow-x: auto !important;    /* keep horizontal scroll if viewport too narrow */
  padding-bottom: 12px;           /* breathing room below */
  margin-bottom: 24px;            /* space under each row heading */
}


  /* 5) Tweak slider labels */
  .stSlider > div > div > label {
    color: #0ff !important;
  }
</style>
""", unsafe_allow_html=True)

df = build_dataset()


df = build_dataset()
if df.empty:
    st.error("❌ No data found. Check your folders/sheets.")
    st.stop()

# ─── 3) Sidebar: selectors & sliders ────────────────────────────────────────────
tickers     = sorted(df["Ticker"].unique())
sel_tickers = st.sidebar.multiselect("🔍 Companies", options=tickers, default=[])
if not sel_tickers:
    st.sidebar.info("Please select at least one company to continue.")
    st.stop()

years_avail  = df[df.Ticker.isin(sel_tickers)]["Year"].dropna().unique()
years_sorted = sorted(int(y) for y in years_avail)
if not years_sorted:
    st.sidebar.error("No years available for the selected companies.")
    st.stop()

sel_year = st.sidebar.slider(
    "📅 Year",
    min_value=years_sorted[0],
    max_value=years_sorted[-1],
    value=years_sorted[-1]
)

st.sidebar.markdown("### 🎛 Simulations")
ebt_adj  = st.sidebar.slider("EBITDA Δ%", -50, 50, 0)
cpx_adj  = st.sidebar.slider("CapEx Δ%",  -50, 50, 0)
debt_adj = st.sidebar.slider("Debt Δ%",   -50, 50, 0)
cash_adj = st.sidebar.slider("Cash Δ%",   -50, 50, 0)
nwc_adj = st.sidebar.slider("NWC Δ%", -50, 50, 0)

# ─── compute the exact historical EV/EBITDA for this year
base = df.query("Year == @sel_year and Ticker in @sel_tickers").copy()
# 1) historical EV and EBITDA
hist_ev     = base["EV"].sum(skipna=True)
hist_ebit   = base["EBITDA"].sum(skipna=True)

# 2) historical net debt = Debt − Cash
hist_net_debt = (
    base["Debt"].sum(skipna=True)
  - base["Cash"].sum(skipna=True)
)

# 3) unlevered EV = EV_hist − NetDebt_hist
unlev_ev    = hist_ev - hist_net_debt

# 4) unlevered multiple
unlev_mult  = (unlev_ev / hist_ebit) if hist_ebit else 0.0


ev_mult_full = (hist_ev / hist_ebit) if hist_ebit else 0

# ─── EV/EBITDA slider with two‐decimal steps and a dynamic key
ev_mult_full = (hist_ev / hist_ebit) if hist_ebit else 0

ev_mult = st.sidebar.slider(
    "EV/EBITDA (unlevered)",
    min_value=0.10,
    max_value=100.00,
    value=round(unlev_mult, 2),   # default to unlevered multiple
    step=0.01,
    key=f"ev_mult_{sel_year}"
)


# ─── 4) Filter & simulate ───────────────────────────────────────────────────────
base = df.query("Year == @sel_year and Ticker in @sel_tickers").copy()
if base.empty:
    st.warning("No data for that selection.")
    st.stop()

sim = base.copy()
for col, pct in [("EBITDA", ebt_adj),
                 ("CapEx",  cpx_adj),
                 ("Debt",   debt_adj),
                 ("Cash",   cash_adj)]:
    sim[col] = sim[col] * (1 + pct / 100)

# apply the NWC % slider BEFORE OCF
sim["ChangeNWC"]      = base["ChangeNWC"] * (1 + nwc_adj / 100)
st.write("DEBUG ChangeNWC:", sim["ChangeNWC"].tolist())

# keep the historical cash taxes constant unless you add a slider for it
sim["CashTaxesPaid"]  = base["CashTaxesPaid"]

# 1) recalc OCF = EBITDA – CashTaxesPaid – ΔNWC (ΔNWC still zero in this simple sim)
sim["OCF"]   =    (      sim["EBITDA"] - sim["CashTaxesPaid"]- sim["ChangeNWC"])

# 2) FCF = OCF – CapEx
sim["FCF"]            = sim["OCF"] - sim["CapEx"]
st.write("🔍 sim FCF after NWC adj:", sim["FCF"].tolist())

# 3) EV and EV/EBITDA 

# recompute sim net‐debt
sim_net_debt = sim["Debt"] - sim["Cash"]

# EV = EBITDA × unlevered‐multiple + change in net debt
sim["EV"] = sim["EBITDA"] * ev_mult + (sim_net_debt - hist_net_debt)

# sim["EV"] = sim["EBITDA"] * ev_mult

sim["EV/EBITDA"]      = sim["EV"] / sim["EBITDA"].replace(0, pd.NA)



# ─── 5) Top metrics: two‐row panels, 5 columns each ────────────────────────────
hist_metrics = [
    ("EBITDA",    "EBITDA",         "$ {:,.0f}"),
    ("CapEx",     "CapEx",          "$ {:,.0f}"),
    ("FCF",       "FCF",            "$ {:,.0f}"),
    ("EV",        "EV",             "$ {:,.0f}"),
    ("EV/EBITDA", "EV/EBITDA",      "{:.2f}x"),
    ("Debt",      "Debt",           "$ {:,.0f}"),
    ("Cash",      "Cash",           "$ {:,.0f}"),
    ("ΔNWC",      "ChangeNWC",      "$ {:,.0f}"),
    ("Interest",  "InterestExpense","$ {:,.0f}"),
]

# first 5 always go on row 1, next 4 on row 2 (with one blank placeholder)
first5 = hist_metrics[:5]
rest4  = hist_metrics[5:]
# pad to length 5
rest5 = rest4 + [(None, None, None)]

for ticker in sel_tickers:
    t_base = base[base.Ticker == ticker]
    if t_base.empty:
        continue
    t_sim = sim[sim.Ticker == ticker]

    st.markdown(f"## {ticker} – Year {sel_year}")

    # ─ Historical Metrics ─────────────────────────────────────────────────────
    st.markdown("### Historical Metrics")
    # Row 1: first 5
    cols = st.columns(5)
    for (label, field, fmt), col in zip(first5, cols):
        if field == "EV/EBITDA":
            total_ebit = t_base["EBITDA"].sum(skipna=True)
            val = (t_base["EV"].sum(skipna=True) / total_ebit) if total_ebit else pd.NA
        else:
            val = t_base[field].sum(skipna=True)
        col.metric(label, fmt.format(val) if pd.notna(val) else "n/a")

    # Row 2: next 4 + blank
    cols = st.columns(5)
    for (label, field, fmt), col in zip(rest5, cols):
        if label is None:
            col.write("")  # placeholder
        else:
            val = t_base[field].sum(skipna=True)
            col.metric(label, fmt.format(val) if pd.notna(val) else "n/a")

    # ─ Simulated Metrics ─────────────────────────────────────────────────────
    st.markdown("### Simulated Metrics")
    # Row 1: first 5
    cols = st.columns(5)
    for (label, field, fmt), col in zip(first5, cols):
        if field == "EV/EBITDA":
            total_ebit = t_base["EBITDA"].sum(skipna=True)
            hist_val   = (t_base["EV"].sum(skipna=True) / total_ebit) if total_ebit else pd.NA
            sim_val    = t_sim["EV/EBITDA"].iat[0]
        else:
            hist_val = t_base[field].sum(skipna=True)
            sim_val  = t_sim[field].iat[0]
        delta = ""
        if pd.notna(hist_val) and pd.notna(sim_val) and hist_val:
            delta = f"{sim_val/hist_val - 1:+.1%}"
        col.metric(label, fmt.format(sim_val) if pd.notna(sim_val) else "n/a", delta,
                   help="FCF = EBITDA − CapEx" if field=="FCF" else "")

    # Row 2: next 4 + blank
    cols = st.columns(5)
    for (label, field, fmt), col in zip(rest5, cols):
        if label is None:
            col.write("")  # placeholder
        else:
            hist_val = t_base[field].sum(skipna=True)
            sim_val  = t_sim[field].iat[0]
            delta = ""
            if pd.notna(hist_val) and pd.notna(sim_val) and hist_val:
                delta = f"{sim_val/hist_val - 1:+.1%}"
            col.metric(label, fmt.format(sim_val) if pd.notna(sim_val) else "n/a", delta,
                       help="FCF = EBITDA − CapEx" if field=="FCF" else "")




# ─── 6) 3D Simulation: EBITDA vs CapEx vs EV ───────────────────────────────────
st.markdown("### 🔭 3D Simulation: EBITDA vs CapEx vs EV")

# 1) build combined DataFrame (must come after sim["FCF"] is recalculated)
plot_df = pd.concat([
    base.assign(Type="Base"),
    sim.assign(Type="Simulated"),
])
plot_df["FCF_mag"]   = plot_df["FCF"].abs().fillna(0)
plot_df["FCF_label"] = plot_df["FCF"].apply(lambda x: "Positive" if x >= 0 else "Negative")

# 2) compute raw min/max for each axis
eb_min, eb_max = plot_df["EBITDA"].min(), plot_df["EBITDA"].max()
cx_min, cx_max = plot_df["CapEx"].min(),  plot_df["CapEx"].max()
ev_min, ev_max = plot_df["EV"].min(),     plot_df["EV"].max()

# 3) add padding so the cube isn’t cramped
pad = 0.05
def padded(rmin, rmax, pad):
    span = rmax - rmin
    if span == 0:
        buffer = abs(rmin) * pad if rmin != 0 else 1
        return [rmin - buffer, rmax + buffer]
    return [rmin - pad * span, rmax + pad * span]
x_range = padded(eb_min, eb_max, pad)
y_range = padded(cx_min, cx_max, pad)
z_range = padded(ev_min, ev_max, 0.001)

# 4) draw the 3D scatter, sizing by the updated FCF_mag
fig3d = px.scatter_3d(
    plot_df,
    x="EBITDA", y="CapEx", z="EV",
    color="FCF_label",
    color_discrete_map={"Negative":"red","Positive":"green"},
    symbol="Type",
    size="FCF_mag",
    size_max=26,                # ↑ bigger max so changes are obvious
    hover_name="Ticker",
    hover_data={
        "Type":      True,
        "EBITDA":    ":.2f",
        "CapEx":     ":.2f",
        "EV":        ":.2f",
        "Debt":      ":.2f",
        "Cash":      ":.2f",
        "EV/EBITDA": ":.2f",
        "FCF":       ":.2f",
        "FCF_mag":   ":.4f",      # show magnitude to 4 decimals
    },
    template="plotly_dark",
    title=f"Year {sel_year}: Base vs Simulated"
)

# 5) add the cube wireframe
import plotly.graph_objects as go
x0, x1 = x_range; y0, y1 = y_range; z0, z1 = z_range
cube_x = [x0,x1,None, x1,x1,None, x1,x0,None, x0,x0,  x0,x1,None, x1,x1,None, x1,x0,None, x0,x0,  x0,x0,None, x1,x1,None, x1,x1,None, x0,x0]
cube_y = [y0,y0,None, y0,y1,None, y1,y1,None, y1,y0,  y0,y0,None, y0,y1,None, y1,y1,None, y1,y0,  y0,y0,None, y1,y1,None, y1,y1,None, y0,y0]
cube_z = [z0,z0,None, z0,z0,None, z0,z0,None, z0,z0,  z1,z1,None, z1,z1,None, z1,z1,None, z1,z1,  z0,z1,None, z0,z1,None, z0,z1,None, z0,z1]
fig3d.add_trace(go.Scatter3d(
    x=cube_x, y=cube_y, z=cube_z,
    mode='lines',
    line=dict(color="rgba(200,200,200,0.3)", width=1),
    showlegend=False
))

# 6) lock axis ranges + view
fig3d.update_layout(
    margin=dict(l=0,r=0,t=40,b=0),
    width=800, height=600,
    uirevision="fixed_view",
    scene=dict(
        aspectmode="cube",
        xaxis=dict(autorange=False, range=x_range, showbackground=True, backgroundcolor="rgba(20,20,20,0.5)"),
        yaxis=dict(autorange=False, range=y_range, showbackground=True, backgroundcolor="rgba(20,20,20,0.5)"),
        zaxis=dict(autorange=False, range=z_range, showbackground=True, backgroundcolor="rgba(20,20,20,0.5)"),
        camera=dict(eye=dict(x=1.8,y=1.4,z=1.2))
    )
)

# 7) render it
st.plotly_chart(fig3d, use_container_width=True)


# ─── 7) EV/EBITDA & FCF Over Time ───────────────────────────────────────────────
st.markdown("### 🔄 EV/EBITDA & FCF Over Time")
time_df = df[df.Ticker.isin(sel_tickers)].copy()
time_df["FCF"] = time_df["EBITDA"] - time_df["CapEx"]
fig2 = px.line(
    time_df, x="Year", y=["FCF","EV/EBITDA"],
    color="Ticker", markers=True,
    template="plotly_dark",
    labels={
        "value":"FCF (USD) / EV/EBITDA (x)",
        "variable":"Metric",
        "Ticker":"Company",
    },
)
st.plotly_chart(fig2, use_container_width=True)

# ─── 8) Data Table ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Data Table")
st.dataframe(
    sim[[
        "Ticker","Year","EBITDA","CapEx",
        "FCF","FCFF","FCFE",
        "ChangeNWC","InterestExpense",
        "EV","EV/EBITDA","Debt","Cash"
    ]],
    use_container_width=True, height=300
)

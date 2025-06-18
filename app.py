# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from scrape_ff5 import get_ff5_data_by_folder
from regression_engine import compute_capm_beta
from regression_engine import compute_ff5_betas
import os
from regression_engine import compute_ff5_betas
import plotly.graph_objects as go



# ─── 1) Streamlit page config ─────────────────────────────────
st.set_page_config(page_title="🚀 Starship Finance Simulator", layout="wide")

# ─── 2) Inject external CSS ────────────────────────────────────
with open("styles.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

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

        pretax = grab_series(xlsx, "Income Statement", r"income (?:before|pre)[ -]tax")

        taxcash = grab_series(xlsx, "Cash Flow", r"income taxes.*paid")

        # compute effective tax rate per year (avoid divide‐by‐zero)
        if pretax and taxcash:
            tax_rate_series = [
                (t / p) if p not in (0, None) else 0.0
                for p, t in zip(pretax, taxcash)
            ]
        else:
          # fallback to zero‐rate if either series is missing
            tax_rate_series = [0.0] * len(years)


      
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
        for i ,(y, e, c, d, ca, v, t, nwc0, ie) in enumerate(zip(
            years, ebitda, capex, debt, cash, ev, taxes_cf,
            change_in_nwc, interest_expense
        )):
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
                "tax_rate":         tax_rate_series[i],
            })

    # build DataFrame
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    
    
# historical ΔDebt & ΔCash
    df["ΔDebt"] = df.groupby("Ticker")["Debt"].diff().fillna(0)
    df["ΔCash"] = df.groupby("Ticker")["Cash"].diff().fillna(0)

# 1) FCFF = EBITDA – CashTaxesPaid – ChangeNWC – CapEx
    df["FCFF"] = (
    df["EBITDA"]
  - df["CashTaxesPaid"]
  - df["ChangeNWC"]
  - df["CapEx"]
)

# 2) FCFE = FCFF – (InterestExpense × (1–tax_rate)) + ΔDebt – ΔCash,
#    using the per‐row tax_rate we built earlier
    df["FCFE"] = (
    df["FCFF"]
  - df["InterestExpense"] * (1 - df["tax_rate"])
  + df["ΔDebt"]
  - df["ΔCash"]
)

    # ── Free Cash Flow (FCF) including ΔNWC ────────────────────────────
    df["FCF"] = (
    df["EBITDA"]
  - df["CashTaxesPaid"]
  - df["ChangeNWC"]
  - df["CapEx"]
)
 

# 3) EV/EBITDA
    df["EV/EBITDA"] = df["EV"] / df["EBITDA"].replace(0, pd.NA)

    return df

df = build_dataset()
if df.empty:
    st.error("❌ No data found. Check your folders/sheets.")
    st.stop()

# ─── 3) Sidebar: selectors & sliders ─────────────────────────────────────────────
tickers     = sorted(df["Ticker"].unique())
sel_tickers = st.sidebar.multiselect("🔎 Companies", options=tickers, default=[])


# Pick a palette and assign each ticker a fixed color:
default_colors = px.colors.qualitative.Plotly
color_map = {
    t: default_colors[i % len(default_colors)]
    for i, t in enumerate(sel_tickers)
}


# ─── Reset only the *removed* tickers from FF-5 state, leave the rest intact ─────────
if "prev_sel_tickers" not in st.session_state:
    st.session_state.prev_sel_tickers = sel_tickers.copy()
elif set(st.session_state.prev_sel_tickers) != set(sel_tickers):
    removed = set(st.session_state.prev_sel_tickers) - set(sel_tickers)
    # remove only those tickers from the two FF-5 dicts:
    for t in removed:
        st.session_state.get("ff5_betas", {}).pop(t, None)
        st.session_state.get("ff5_errors", {}).pop(t, None)
        st.session_state.get("ff5", {}).pop(t, None)
    # update our record
    st.session_state.prev_sel_tickers = sel_tickers.copy()


# ─── 4) Build shared color map ────────────────────────────────────────────────

default_colors = px.colors.qualitative.Plotly
color_map = {
    t: default_colors[i % len(default_colors)]
    for i, t in enumerate(sel_tickers)
}

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
st.sidebar.markdown(
    f"ℹ️ Note: some tickers (e.g. AIR.NZ) only have data through 2024. "
    "Move the slider down to see their multiples."
)

########################
    # ─── 3a) Choose Estimation Method ─────────────────────────────────────────────
method = st.sidebar.radio(
        "⚙️ Estimation Method",
        options=["Historical", "FF-5", "CAPM"],
        index=0,
    )


def ticker_to_region(ticker: str) -> str:
    parts = ticker.split(".")
    if len(parts) == 1:
        # no exchange suffix → default to US
        return "US"

    suffix = parts[-1].upper()
    # map suffixes to your folder names
    region_map = {
        "AX": "AU",   # Australian tickers
        "NZ": "NZ",   # New Zealand tickers
        "DE": "DE",   # German tickers
        "O":  "US",   # .O suffix → US
    }
    return region_map.get(suffix, "US")



if method == "FF-5":
        st.sidebar.markdown("### 🔢 Download FF-5 Factors")
        if st.sidebar.button("📥 Fetch FF-5 Data"):
            ff5_results: dict[str, pd.DataFrame] = {}
            for ticker in sel_tickers:
                folder = ticker_to_region(ticker)
                with st.spinner(f"Downloading FF-5 for {ticker}…"):
                    try:
                        df_ff5 = get_ff5_data_by_folder(ticker, folder)
                        ff5_results[ticker] = df_ff5
                    except Exception as e:
                        st.sidebar.error(f"❌ {ticker}: {e}")
            if ff5_results:
                st.sidebar.success("✅ All FF-5 data downloaded")
                st.session_state["ff5"] = ff5_results
                
import os
import plotly.graph_objects as go

# ─── 3b) Compute & display FF-5 betas ──────────────────────────────────────────
if method == "FF-5" and "ff5" in st.session_state:
    # pull the raw FF-5 Data dict out of session
    ff5_results = st.session_state["ff5"]

    # now compare the ticker sets to decide whether to recompute
    if set(ff5_results) != set(st.session_state.get("ff5_betas", {})):
        # Step 1: pull in & annualize FF-5 risk-free rate & market premium
        sample_ff5     = next(iter(ff5_results.values()))
        sample_ff5     = sample_ff5.apply(lambda col: pd.to_numeric(col, errors="coerce"))
        monthly_rf     = sample_ff5["RF"].mean()
        monthly_mktrf  = sample_ff5["Mkt-RF"].mean()
        rf_annual      = (1 + monthly_rf)   ** 12 - 1
        mktprem_annual = (1 + monthly_mktrf) ** 12 - 1

        # Step 2: compute a fresh set of betas + errors
        betas_by_ticker  = {}
        errors_by_ticker = {}

        for ticker, ff5_df in ff5_results.items():
            # load the pre-saved returns CSV
            path = f"attached_assets/returns_{ticker.replace('.', '_')}.csv"
            if not os.path.isfile(path):
                st.sidebar.error(f"Missing returns file for {ticker}: {path}")
                continue

            returns_df = pd.read_csv(path, parse_dates=True, index_col=0)
            stock_ret  = returns_df["Return"]

            # compute the five betas + residuals
            res = compute_ff5_betas(stock_ret, ff5_df)
            betas_by_ticker[ticker] = {
                "Mkt-RF": res["market_beta"],
                "SMB":    res["smb_beta"],
                "HML":    res["hml_beta"],
                "RMW":    res["rmw_beta"],
                "CMA":    res["cma_beta"],
            }
            errors_by_ticker[ticker] = {
                "r_squared": res["r_squared"],
                "alpha":     res["alpha"],
            }

        # stash them in session_state
        st.session_state["ff5_betas"]  = betas_by_ticker
        st.session_state["ff5_errors"] = errors_by_ticker

# ─── 3c) CAPM regression ─────────────────────────────────────────────────────────
elif method == "CAPM":
    st.sidebar.markdown("### 📈 Run CAPM Regression")
    if st.sidebar.button("📥 Fetch Returns & Compute β"):
        capm_results: dict[str, dict[str, float]] = {}
        for ticker in sel_tickers:
            with st.spinner(f"Fetching returns and regressing CAPM for {ticker}…"):
                try:
                    beta, error = compute_capm_beta(ticker)
                    capm_results[ticker] = {"beta": beta, "error": error}
                except Exception as e:
                    st.sidebar.error(f"❌ {ticker}: {e}")
        if capm_results:
            st.sidebar.success("✅ CAPM betas computed")
            st.session_state["capm"] = capm_results
#########

st.sidebar.markdown("### 🎛 Simulations")
ebt_adj  = st.sidebar.slider("EBITDA Δ%", -50, 50, 0)
cpx_adj  = st.sidebar.slider("CapEx Δ%",  -50, 50, 0)
debt_adj = st.sidebar.slider("Debt Δ%",   -50, 50, 0)
cash_adj = st.sidebar.slider("Cash Δ%",   -50, 50, 0)
nwc_adj  = st.sidebar.slider("NWC Δ%", -50, 50, 0)

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
# adjust everything except Cash multiplicatively
for col, pct in [
    ("EBITDA", ebt_adj),
    ("CapEx",  cpx_adj),
    ("Debt",   debt_adj),
]:
    sim[col] = sim[col] * (1 + pct / 100)

# now adjust Cash so +% always moves the balance toward +∞
sim["Cash"] = base["Cash"] + base["Cash"].abs() * (cash_adj / 100)


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
    ("Tax Rate",  "tax_rate",    "{:.1%}"), 
]

# first 5 always go on row 1, next 4 on row 2 (with one blank placeholder)
# two rows of 5 metrics each
first5 = hist_metrics[:5]
rest5  = hist_metrics[5:10]    # exactly the other five


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
st.plotly_chart(fig3d, use_container_width=True, key="ev_cube_chart")


# ─── 7) EV/EBITDA & FCF Over Time ───────────────────────────────────────────────
st.markdown("### 🔄 EV/EBITDA & FCF Over Time")
time_df = df[df.Ticker.isin(sel_tickers)].copy()
time_df["FCF"] = time_df["EBITDA"] - time_df["CapEx"]
fig2 = px.line(
    time_df, x="Year", y=["FCF","EV/EBITDA"],
    color="Ticker", color_discrete_map=color_map, markers=True,
    template="plotly_dark",
    labels={
        "value":"FCF (USD) / EV/EBITDA (x)",
        "variable":"Metric",
        "Ticker":"Company",
    },
)
st.plotly_chart(fig2, use_container_width=True, key="ev_cube_chart2")

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

# ─── 9 & 10) FF-5 Betas, Errors & WACC (always render if there’s at least one β) ─────────────────────
if method == "FF-5":
    # 9-0) grab all cached betas/errors
    all_betas  = st.session_state.get("ff5_betas", {})
    all_errors = st.session_state.get("ff5_errors", {})

    # 9-1) filter for the tickers the user actually selected
    betas  = {t: all_betas[t]  for t in sel_tickers if t in all_betas}
    errors = {t: all_errors[t] for t in sel_tickers if t in all_errors}

    # 9-2) Regression‐errors table
    if errors:
        st.markdown("#### FF-5 Regression Errors")
        df_err = pd.DataFrame.from_dict(errors, orient="index")
        st.dataframe(df_err.style.format({
            "r_squared":"{:.2f}",
            "alpha":    "{:.4f}",
        }))

    # 9-3) β-chart (use same colors as your other plots)
    if betas:
        st.markdown("#### FF-5 Factor Betas")
        fig = go.Figure()
        for ticker, bdict in betas.items():
            fig.add_trace(go.Scatter(
                x=list(bdict.keys()),
                y=list(bdict.values()),
                mode="lines+markers",
                name=ticker,
                line  = dict(color=color_map[ticker]),
                marker= dict(color=color_map[ticker]),
            ))
        fig.update_layout(
            title="FF-5 Factor Betas",
            xaxis_title="Factor",
            yaxis_title="β Coefficient",
            legend_title="Ticker",
        )
        st.plotly_chart(fig, use_container_width=True, key="ff5_betas_chart")

    # 10) WACC by company (only if there’s at least one β)
    if betas:
        # 10-0) annualize RF & market premium
        ff5_results    = st.session_state["ff5"]
        sample_ff5     = next(iter(ff5_results.values()))
        sample_ff5     = sample_ff5.apply(lambda col: pd.to_numeric(col, errors="coerce"))
        monthly_rf     = sample_ff5["RF"].mean()
        monthly_mktrf  = sample_ff5["Mkt-RF"].mean()
        rf_annual      = (1 + monthly_rf)   ** 12 - 1
        mktprem_annual = (1 + monthly_mktrf) ** 12 - 1

        # compute WACC
        wacc_rows = []
        for t in betas.keys():               # only loop tickers for which you have betas
            sub = df.query("Ticker == @t and Year == @sel_year")
            if sub.empty:
                continue
            row = sub.iloc[0]

            β     = betas[t]["Mkt-RF"]
            Re    = rf_annual + β * mktprem_annual
            Rd    = abs(row["InterestExpense"]) / row["Debt"] if row["Debt"] else 0.0
            Rd_at = Rd * (1 - row["tax_rate"])
            netD  = row["Debt"] - row["Cash"]
            E_val = row["EV"] - netD
            tot   = (E_val + netD) or 1.0
            wE    = E_val / tot
            wD    = netD / tot
            wacc  = wE * Re + wD * Rd_at

            wacc_rows.append({
                "Ticker":   t,
                "Re (%)":   f"{Re*100:.2f}",
                "Rdₐₜ (%)": f"{Rd_at*100:.2f}",
                "wE (%)":   f"{wE*100:.1f}",
                "wD (%)":   f"{wD*100:.1f}",
                "WACC (%)": f"{wacc*100:.2f}",
            })

        if wacc_rows:
            wacc_df = pd.DataFrame(wacc_rows).set_index("Ticker")
            st.markdown("#### 🧮 WACC by Company")
            st.dataframe(wacc_df, width= 0)
# ── end FF-5 Betas, Errors & WACC block ─────────────────────────────────





# ─── 10) Footer ────────────────────────────────────────────────────────────
st.markdown(
    """
    *FF-5 factor betas data courtesy of the [Kenneth R. French Data Library](
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/index.html).*

    Github: https://github.com/tomektomeknyc/companies

    """,
    unsafe_allow_html=True,
)



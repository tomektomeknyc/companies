# finance_data_loader.py
import pandas as pd
from pathlib import Path
import re

# ─── 1) Point to your file ─────────────────────────────────────────────────────
FILE     = Path("AU") / "CSL.AX.xlsx"

# ─── 2) Constants ──────────────────────────────────────────────────────────────
YEAR_ROW = 10                 # zero-based (Excel row 11)
COLS     = list(range(1, 16)) # B–P → indices 1–15

# ─── 3) Raw loader ─────────────────────────────────────────────────────────────
def load_raw(sheet_name: str):
    df = pd.read_excel(
        FILE,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl"
    )
    years = df.iloc[YEAR_ROW, COLS].astype(str).tolist()
    return df, years

# ─── 4) Grab by regex ──────────────────────────────────────────────────────────
def grab_series(sheet_name: str, pattern: str):
    df, years = load_raw(sheet_name)
    col0 = df.iloc[:, 0].astype(str).str.lower().str.strip()
    mask = col0.str.contains(pattern, regex=True, na=False)
    if mask.any():
        row = df.loc[mask, :].iloc[0]
        return row.iloc[COLS].tolist()
    return [pd.NA] * len(years)

# ─── 5) Pretty-print a series ───────────────────────────────────────────────────
def print_series(name: str, years: list[str], vals: list):
    print(f"\n{name} {years[0]}–{years[-1]}:")
    for y, v in zip(years, vals):
        print(f"  {y}: {v}")

# ─── 6) Grab everything ───────────────────────────────────────────────────────
_, years    = load_raw("Income Statement")

ebitda_vals = grab_series("Income Statement", r"earnings before.*ebitda")
da_vals     = grab_series("Income Statement", r"depreciation.*amortization")
capex_vals  = grab_series("Cash Flow",           r"capital expenditure|capex")
debt_vals   = grab_series("Balance Sheet",       r"total debt|debt\b")
cash_vals   = grab_series("Balance Sheet",       r"cash and cash equivalents|cash$")
ev_vals     = grab_series("Financial Summary",   r"^enterprise value\s*$")

# ←–––––––––––––– FIXED EV/EBITDA regex –––––––––––––––––––––––––––––––––––––––––
evebitda_vals = grab_series(
    "Valuation",
    # allow the full "enterprise value to earnings before interest, taxes, depreciation & amortization"
    r"\b(?:ev\s*(?:/|to)\s*ebitda|enterprise value to earnings(?: before.*)?)\b"
)
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# ─── 7) Print them all ─────────────────────────────────────────────────────────
print_series("EBITDA",                    years, ebitda_vals)
print_series("Depreciation & Amortization", years, da_vals)
print_series("CapEx",                      years, capex_vals)
print_series("Total Debt",                 years, debt_vals)
print_series("Cash",                       years, cash_vals)
print_series("Enterprise Value",           years, ev_vals)
print_series("EV/EBITDA",                  years, evebitda_vals)

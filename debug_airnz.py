#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import re
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────────
# adjust the subfolder + filename as needed:
test = Path(__file__).parent / "NZ" / "AIR.NZ.xlsx"

# ─── YEAR-ROW DETECTOR ────────────────────────────────────────────────────────────
def detect_year_row(df: pd.DataFrame, max_col=20):
    """
    Scan each row; treat a cell as a "year" if it's an integer 1900–2100
    Return the first row index where >= 3 such year-cells appear within the first max_col columns.
    """
    for idx, row in df.iterrows():
        count = 0
        for val in row.iloc[:max_col]:
            try:
                y = int(val)
                if 1900 <= y <= 2100:
                    count += 1
            except Exception:
                continue
        if count >= 3:
            return idx
    return None

# ─── 1) CAN WE OPEN THE FILE? ──────────────────────────────────────────────────────
print(f"\n\n── DEBUG: {test!s} ────────────────────────────────────────")
if not test.exists():
    print("❌ File does not exist at:", test)
    sys.exit(1)

try:
    xl = pd.ExcelFile(test, engine="openpyxl")
    print("✔️  Available sheets:", xl.sheet_names)
except Exception as e:
    print("❌ Could not open workbook:", e)
    sys.exit(1)

# ─── 2) FOR EACH SHEET: DETECT YEAR ROW & SHOW YEARS ───────────────────────────────
for sheet in xl.sheet_names:
    print(f"\n––– Sheet: {sheet!r} ––––––––––––––––––––––––––––––––––––––––––")
    try:
        df = pd.read_excel(test, sheet_name=sheet, header=None, engine="openpyxl")
    except Exception as e:
        print("   ❌ Failed to read sheet:", e)
        continue

    yr = detect_year_row(df)
    if yr is None:
        print("   ⚠️  No obvious year-row found (needs ≥3 ints 1900–2100).")
        continue

    print(f"   ✔️  Detected YEAR_ROW = {yr}")
    # show the first 15 columns of that row:
    labels = df.iloc[yr, :15].tolist()
    print("   → row values:", labels)

    # BONUS: run your EBITDA regex against col A of this sheet
    col0 = df.iloc[:, 0].astype(str).str.lower().str.strip()
    pat = re.compile(r"earnings before.*ebitda", flags=re.IGNORECASE)
    matches = col0[col0.str.contains(pat, na=False)]
    if not matches.empty:
        print("   🔍 EBITDA hits at rows:", matches.index.tolist())
        print("      sample text:", matches.iloc[0])
    else:
        print("   🔍 No EBITDA match in column A.")

print("\n\nDone.")

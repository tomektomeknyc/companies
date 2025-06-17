# scrape_ff5.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from io import StringIO
from pathlib import Path
import difflib


FF5_CACHE_DIR = Path("ff5_cache")
FF5_CACHE_DIR.mkdir(exist_ok=True)

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Determine FF5 Region Label from Folder Name
# ─────────────────────────────────────────────────────────────────────────────
def get_ff5_label_and_url(folder_name: str) -> tuple[str, str]:
    region_to_ff5 = {
        "US": (
            "Fama/French Developed 5 Factors",
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_5_Factors_CSV.zip"
        ),
        "DE": (
            "Fama/French Europe 5 Factors",
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_CSV.zip"
        ),
        "NZ": (
            "Fama/French Asia Pacific ex Japan 5 Factors",
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_CSV.zip"
        ),
        "AU": (
            "Fama/French Asia Pacific ex Japan 5 Factors",
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_CSV.zip"
        ),
    }

    try:
        label, url = region_to_ff5[folder_name.upper()]
    except KeyError:
        raise ValueError(f"No FF5 mapping found for region folder: {folder_name}")

    return label, url

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fetch and Parse HTML Page
# ─────────────────────────────────────────────────────────────────────────────
def get_soup():
    response = requests.get(BASE_URL)
    if not response.ok:
        raise RuntimeError(f"Failed to fetch HTML from {BASE_URL}")
    return BeautifulSoup(response.text, "html.parser")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract CSV URL for a Given FF5 Label
# ─────────────────────────────────────────────────────────────────────────────
def extract_csv_url(label: str) -> str:
    soup = get_soup()
    for p in soup.find_all("p"):
        if label in p.get_text():
            links = p.find_all("a", href=True)
            for link in links:
                if link.text.strip().upper() == "CSV":
                    return requests.compat.urljoin(BASE_URL, link["href"])
    raise RuntimeError(f"No CSV found for label '{label}'")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Download and Cache CSV
# ─────────────────────────────────────────────────────────────────────────────
def download_and_cache_csv(ticker: str, label: str, csv_url: str) -> pd.DataFrame:
    from zipfile import ZipFile
    from io import BytesIO, StringIO
    import re

    response = requests.get(csv_url)
    if not response.ok:
        raise RuntimeError("Failed to download CSV ZIP")

    z = ZipFile(BytesIO(response.content))
    csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]
    if not csv_files:
        raise RuntimeError("No CSV file found inside ZIP")

    csv_filename = csv_files[0]
    content = z.read(csv_filename).decode("utf-8")
    lines = content.splitlines()

    # Detect header and data rows
    header_index = next(i for i, line in enumerate(lines) if "Mkt-RF" in line)
    try:
        data_end = next(i for i, line in enumerate(lines[header_index:]) if "Copyright" in line)
        data_end += header_index
    except StopIteration:
        data_end = len(lines)

    # Check if header line has proper CSV format or is space-separated
    header_line = lines[header_index].strip()
    if "," not in header_line and " " in header_line:
        # Space-separated header - need to reformat
        header_parts = header_line.split()
        # Create proper CSV header
        csv_header = ",".join(header_parts)
        # Replace the header line
        lines[header_index] = csv_header
    
    # Read full block (header + data)
    all_lines = "\n".join(lines[header_index:data_end])
    df = pd.read_csv(StringIO(all_lines), skipinitialspace=True, sep=None, engine='python')

    # Rename and set index
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    
    # Clean and filter valid date entries
    df["Date"] = df["Date"].astype(str).str.strip()
    df = df[df["Date"].str.match(r'^\d{6}$')]  # Keep only 6-digit dates (YYYYMM)
    
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df.set_index("Date", inplace=True)

    # Save
    filename = f"{ticker}_{label.replace('/', '_').replace(' ', '_')}.csv"
    filepath = FF5_CACHE_DIR / filename
    df.to_csv(filepath)
    print(f"💾 Saved FF5 CSV to: {filepath}")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 5. Wrapper Function: Ticker + Folder → FF5 DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def get_ff5_data_by_folder(ticker: str, folder: str) -> pd.DataFrame:
    label, csv_url = get_ff5_label_and_url(folder)
    print(f"\n🧠 FF5 Label: {label}")
    print(f"🔗 CSV URL: {csv_url}")

    df = download_and_cache_csv(ticker, label, csv_url)

    # Display summary information
    print(f"\n📊 Dataset Summary:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 6. Run As Script (for testing)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ticker = "CBA.AU"
    folder = "AU"  # Test with European data
    df_ff5 = get_ff5_data_by_folder(ticker, folder)
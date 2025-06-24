import streamlit as st
import sys

st.title("🔍 Import Diagnostics")
st.write("Testing all imports on Streamlit Cloud environment...")

# Show Python version info
st.subheader("Environment Info")
st.write(f"**Python Version:** {sys.version}")
st.write(f"**Streamlit Version:** {st.__version__}")

st.subheader("Import Test Results")

def test_import(module_name, import_statement, description=""):
    """Test a single import and display result"""
    try:
        exec(import_statement)
        st.success(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        st.error(f"❌ {module_name} FAILED: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ {module_name} ERROR: {str(e)}")
        return False

# Test core packages
st.write("**Core Data Science Packages:**")
test_import("pandas", "import pandas as pd", "Data manipulation")
test_import("numpy", "import numpy as np", "Numerical computing")
test_import("scipy", "import scipy", "Scientific computing")

st.write("**Machine Learning:**")
test_import("scikit-learn", "from sklearn.linear_model import LinearRegression", "ML algorithms")

st.write("**Visualization:**")
test_import("plotly", "import plotly.express as px", "Interactive plots")
test_import("plotly.graph_objects", "import plotly.graph_objects as go", "Advanced plotting")

st.write("**File Processing:**")
test_import("openpyxl", "import openpyxl", "Excel file handling")
test_import("beautifulsoup4", "from bs4 import BeautifulSoup", "HTML parsing")

st.write("**HTTP & Environment:**")
test_import("requests", "import requests", "HTTP requests")
test_import("python-dotenv", "from dotenv import load_dotenv", "Environment variables")

st.write("**Built-in Python Modules:**")
test_import("pathlib", "from pathlib import Path", "Path handling")
test_import("os", "import os", "Operating system interface")
test_import("warnings", "import warnings", "Warning control")

st.write("**Custom Project Modules:**")
test_import("scrape_ff5", "from scrape_ff5 import get_ff5_data_by_folder", "FF5 data scraping")
test_import("regression_engine", "from regression_engine import compute_capm_beta, compute_ff5_betas", "Regression calculations")
test_import("stock_returns", "from stock_returns import fetch_daily_returns", "Stock data fetching")
test_import("finance_data_loader", "from finance_data_loader import *", "Finance data utilities")

st.subheader("Instructions")
st.info("""
**If you see red ❌ errors:**
1. Copy the exact error messages
2. Paste them in our chat
3. I'll give you specific fixes for requirements.txt

**This diagnostic tool helps avoid 24-hour debugging sessions!**
""")

st.subheader("Current Requirements.txt")
st.code("""
streamlit
pandas
numpy
scipy
scikit-learn
plotly
openpyxl
beautifulsoup4
requests
python-dotenv
""", language="text")

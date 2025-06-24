# regression_engine.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def compute_capm_beta(stock_returns: pd.Series, ff5_data: pd.DataFrame):
    # Align data
    aligned_data = pd.concat([stock_returns, ff5_data['Mkt-RF']], axis=1, join='inner')
    aligned_data = aligned_data.dropna()
    
    if len(aligned_data) < 10:
        return None, None, None
    
    # Prepare data for regression
    X = aligned_data['Mkt-RF'].values.reshape(-1, 1)
    y = aligned_data.iloc[:, 0].values  # stock returns
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    beta = model.coef_[0]
    alpha = model.intercept_
    r_squared = model.score(X, y)
    
    return alpha, beta, r_squared

def compute_ff5_betas(stock_returns: pd.Series, ff5_data: pd.DataFrame):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Align the data
    aligned_data = pd.concat([stock_returns, ff5_data], axis=1, join='inner')
    aligned_data = aligned_data.dropna()
    
    if len(aligned_data) < 50:  # Need enough data points
        return {
            "market_beta": None,
            "smb_beta": None, 
            "hml_beta": None,
            "rmw_beta": None,
            "cma_beta": None,
            "alpha": None,
            "r_squared": None
        }
    
    # Calculate excess returns
    stock_col = aligned_data.columns[0]  # First column is stock returns
    excess_stock = aligned_data[stock_col] - aligned_data['RF']
    
    # Prepare factors
    X = aligned_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values
    y = excess_stock.values
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    return {
        "market_beta": float(model.coef_[0]),
        "smb_beta": float(model.coef_[1]), 
        "hml_beta": float(model.coef_[2]),
        "rmw_beta": float(model.coef_[3]),
        "cma_beta": float(model.coef_[4]),
        "alpha": float(model.intercept_),
        "r_squared": float(model.score(X, y))
    }

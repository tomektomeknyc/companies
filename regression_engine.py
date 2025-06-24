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

def compute_ff5_betas(stock_returns: pd.Series, ff5_data: pd.DataFrame) -> dict:
    # 1) Ensure all factor columns are numeric
    factors = ff5_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # 2) Align on common dates
    common_idx = stock_returns.index.intersection(factors.index)
    stock_ret_aligned = stock_returns.loc[common_idx].astype(float)
    factors_aligned = factors.loc[common_idx]

    # 3) Compute excess returns
    excess_stock = stock_ret_aligned - factors_aligned["RF"]

    # 4) Run the multivariate regression
    X = factors_aligned[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    y = excess_stock
    model = sm.OLS(y, sm.add_constant(X)).fit()

    return {
        "market_beta": float(model.params["Mkt-RF"]),
        "smb_beta":    float(model.params["SMB"]),
        "hml_beta":    float(model.params["HML"]),
        "rmw_beta":    float(model.params["RMW"]),
        "cma_beta":    float(model.params["CMA"]),
        "alpha":       float(model.params["const"]),
        "r_squared":   float(model.rsquared),
        "residuals":   model.resid.tolist(),
        "dates":       factors_aligned.index.strftime("%Y-%m").tolist(),
    }

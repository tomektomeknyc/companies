# regression_engine.py
import pandas as pd
import statsmodels.api as sm


def compute_capm_beta(stock_returns: pd.Series, ff5_data: pd.DataFrame) -> dict:
    aligned = ff5_data.copy()
    aligned = aligned.loc[stock_returns.index]
    excess_stock = stock_returns - aligned["RF"]

    X = aligned[["Mkt-RF"]]
    y = excess_stock

    model = sm.OLS(y, sm.add_constant(X)).fit()

    return {
        "market_beta": float(model.params["Mkt-RF"]),
        "alpha": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "residuals": model.resid.tolist(),
        "dates": aligned.index.strftime("%Y-%m").tolist()
    }


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

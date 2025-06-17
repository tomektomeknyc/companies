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
    aligned = ff5_data.copy()
    aligned = aligned.loc[stock_returns.index]
    excess_stock = stock_returns - aligned["RF"]

    X = aligned[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    y = excess_stock

    model = sm.OLS(y, sm.add_constant(X)).fit()

    return {
        "market_beta": float(model.params["Mkt-RF"]),
        "smb_beta": float(model.params["SMB"]),
        "hml_beta": float(model.params["HML"]),
        "rmw_beta": float(model.params["RMW"]),
        "cma_beta": float(model.params["CMA"]),
        "alpha": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "residuals": model.resid.tolist(),
        "dates": aligned.index.strftime("%Y-%m").tolist()
    }
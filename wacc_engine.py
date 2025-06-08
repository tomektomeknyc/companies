# wacc_engine.py
import numpy as np
import pandas as pd

def cost_of_equity(
    ex_stock: np.ndarray,
    ex_mkt:   np.ndarray,
    rf:       np.ndarray,
    factors:  dict | None = None,
):
    """
    Return (beta_market, Re_series).
    If `factors` is None → CAPM.
    Else `factors` must contain SMB,HML,RMW,CMA → Fama-French-5.
    All inputs are 1-D numpy arrays of equal length (monthly or annual excess returns).
    """
    y = ex_stock

    if factors is None:           # ─ CAPM
        X = np.column_stack([ex_mkt, np.ones_like(y)])
        beta, _alpha = np.linalg.lstsq(X, y, rcond=None)[0]
        re = rf + beta * ex_mkt
        return float(beta), pd.Series(re)

    # ─ FF-5
    X = np.column_stack([
        ex_mkt,
        factors["SMB"],
        factors["HML"],
        factors["RMW"],
        factors["CMA"],
        np.ones_like(y),
    ])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta_mkt = float(coefs[0])
    re = (
        rf
        + beta_mkt * ex_mkt
        + coefs[1] * factors["SMB"]
        + coefs[2] * factors["HML"]
        + coefs[3] * factors["RMW"]
        + coefs[4] * factors["CMA"]
    )
    return beta_mkt, pd.Series(re)

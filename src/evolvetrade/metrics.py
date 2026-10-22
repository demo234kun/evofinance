import numpy as np
import pandas as pd


def evaluate(returns: pd.Series, annualization: int = 252) -> dict:
    r = returns.fillna(0.0).astype(float)
    equity = (1.0 + r).cumprod()
    vol = r.std(ddof=1) * np.sqrt(annualization)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(annualization) if r.std(ddof=1) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "sharpe": float(sharpe),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(-drawdown.min()),
        "win_rate": float((r > 0).mean()),
        "volatility": float(vol),
    }


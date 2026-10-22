from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(config: dict, output: Path) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"])
    n = config["trading_days"]
    assets = config["assets"]
    dates = pd.bdate_range("2024-01-02", periods=n)
    market = np.r_[np.full(n // 3, 0.0005), np.full(n // 3, -0.00035), np.full(n - 2 * (n // 3), 0.0007)]
    common = market + rng.normal(0, 0.007, n)
    returns = {}
    for i, asset in enumerate(assets):
        beta = 0.7 + 0.6 * (i / max(len(assets) - 1, 1))
        alpha = (i % 5 - 2) * 0.00004
        returns[asset] = alpha + beta * common + rng.normal(0, 0.009 + 0.0005 * (i % 4), n)
    ret = pd.DataFrame(returns, index=dates).clip(-0.18, 0.18)
    prices = 100.0 * (1.0 + ret).cumprod()
    prices.index.name = "date"
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output)
    return prices


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if prices.index.has_duplicates or prices.columns.duplicated().any():
        raise ValueError("duplicate date or asset")
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("invalid prices")
    return prices.astype(float)


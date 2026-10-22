import numpy as np
import pandas as pd


def _paper_exposure(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.fillna(False).astype(float) / signal.shape[1]


def baseline_weights(prices: pd.DataFrame, method: str) -> pd.DataFrame:
    returns = prices.pct_change()
    if method == "B&H":
        return pd.DataFrame(1 / prices.shape[1], index=prices.index, columns=prices.columns)
    if method == "SPY":
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w.iloc[:, 0] = 1.0
        return w
    if method == "SMA":
        return _paper_exposure(prices.rolling(20).mean() > prices.rolling(60).mean())
    if method == "MACD":
        macd = prices.ewm(span=12, adjust=False).mean() - prices.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        return _paper_exposure(macd > signal)
    if method == "ZMR":
        z = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
        return _paper_exposure(z < -1.0)
    if method == "KDJ&RSI":
        low = prices.rolling(9).min()
        high = prices.rolling(9).max()
        k = (prices - low) / (high - low + 1e-12)
        gain = returns.clip(lower=0).rolling(14).mean()
        loss = -returns.clip(upper=0).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-12))
        return _paper_exposure((k < 0.2) & (rsi < 35))
    raise ValueError(method)


def backtest_weights(prices: pd.DataFrame, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    asset_returns = prices.pct_change().fillna(0.0)
    applied = weights.shift(1).reindex_like(asset_returns).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    return (applied * asset_returns).sum(axis=1) - turnover * cost_bps / 10000.0


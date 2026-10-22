import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.baselines import baseline_weights, backtest_weights
from evolvetrade.metrics import evaluate

config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
prices = pd.read_csv(ROOT / "data" / "real" / "us_daily_adjusted_close.csv", index_col=0, parse_dates=True)
rows = []
for period, (start, end) in config["evaluation_windows"].items():
    window = prices.loc[:end]
    evaluation_index = prices.loc[start:end].index
    for method in ["SPY", "B&H", "MACD", "KDJ&RSI", "ZMR", "SMA"]:
        if method == "SPY":
            gross = prices["SPY"].pct_change().reindex(evaluation_index).fillna(0)
            turnover = 0.0
        else:
            universe = window[config["assets"]]
            weights = baseline_weights(universe, method)
            gross = backtest_weights(universe, weights, 0).reindex(evaluation_index).fillna(0)
            turnover = float((0.5 * weights.diff().abs().sum(axis=1)).reindex(evaluation_index).fillna(0).sum())
        net = gross - turnover / max(len(gross), 1) * config["transaction_cost_bps"] / 10000
        rows.append({"method": method, "period": period, **evaluate(gross), **{f"net_{k}": v for k, v in evaluate(net).items()}, "turnover": turnover})
result = pd.DataFrame(rows)
result.to_csv(ROOT / "result" / "rule_baselines.csv", index=False)
print(result.to_string(index=False))


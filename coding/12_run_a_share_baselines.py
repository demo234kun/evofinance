import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.baselines import baseline_weights, backtest_weights
from evolvetrade.metrics import evaluate

cfg = json.loads((ROOT / "config" / "a_share_config.json").read_text(encoding="utf-8"))
prices = pd.read_csv(ROOT / "data" / "a_share" / "a_share_daily_adjusted_close.csv", index_col=0, parse_dates=True)
assets = pd.read_csv(ROOT / "data" / "a_share" / "selected_assets.csv")["symbol"].tolist()
rows = []
for period, (start, end) in cfg["evaluation_windows"].items():
    evaluation_index = prices.loc[start:end].index
    history = prices.loc[:end, assets]
    for method in ["CSI300_EW", "B&H", "MACD", "KDJ&RSI", "ZMR", "SMA"]:
        if method == "CSI300_EW":
            gross = prices["CSI300_EW"].pct_change(fill_method=None).reindex(evaluation_index).fillna(0)
            turnover = 0.0
        else:
            weights = baseline_weights(history, method)
            gross = backtest_weights(history, weights, 0).reindex(evaluation_index).fillna(0)
            turnover = float((0.5 * weights.diff().abs().sum(axis=1)).reindex(evaluation_index).fillna(0).sum())
        net = gross - turnover / max(len(gross), 1) * cfg["transaction_cost_bps"] / 10000
        rows.append({"method": method, "period": period, **evaluate(gross),
                     **{f"net_{key}": value for key, value in evaluate(net).items()}, "turnover": turnover})
pd.DataFrame(rows).to_csv(ROOT / "result" / "a_share_rule_baselines.csv", index=False, encoding="utf-8-sig")

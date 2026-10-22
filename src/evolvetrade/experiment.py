import json
from pathlib import Path
import pandas as pd

from .baselines import backtest_weights, baseline_weights
from .metrics import evaluate
from .policy import Policy, PolicyAgent, allocations


def run_agent(prices: pd.DataFrame, config: dict, evolving: bool, update_interval: int | None = None) -> tuple[pd.Series, list[dict], pd.DataFrame]:
    interval = update_interval or config["update_interval"]
    policy = Policy(cash_floor=config["min_cash_weight"])
    updater = PolicyAgent()
    returns = prices.pct_change().fillna(0.0)
    weights = pd.DataFrame(0.0, index=prices.index, columns=list(prices.columns) + ["CASH"])
    records: list[dict] = []
    batch_returns: list[float] = []
    batch_trace: list[dict] = []
    for i, date in enumerate(prices.index):
        if i < config["lookback_long"]:
            weights.loc[date, "CASH"] = 1.0
            continue
        w, trace = allocations(prices.iloc[: i + 1], policy, config["max_asset_weight"])
        weights.loc[date] = w.reindex(weights.columns).fillna(0.0)
        realized = 0.0 if i + 1 >= len(prices) else float((w.drop("CASH") * returns.iloc[i + 1]).sum())
        record = {
            "decision_date": str(date.date()),
            "feedback_date": None if i + 1 >= len(prices) else str(prices.index[i + 1].date()),
            "policy": policy.to_dict(),
            "allocation": {k: float(v) for k, v in w.items()},
            "trace": trace,
            "realized_return": realized,
        }
        records.append(record)
        batch_returns.append(realized)
        batch_trace.append(trace)
        if evolving and len(batch_returns) == interval and i + 1 < len(prices):
            policy = updater.refine(policy, pd.Series(batch_returns), batch_trace)
            batch_returns, batch_trace = [], []
    asset_weights = weights.drop(columns="CASH")
    portfolio_returns = backtest_weights(prices, asset_weights, config["transaction_cost_bps"])
    return portfolio_returns, records, weights


def run_experiment(prices: pd.DataFrame, config: dict, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["SPY", "B&H", "MACD", "KDJ&RSI", "ZMR", "SMA"]:
        r = backtest_weights(prices, baseline_weights(prices, name), config["transaction_cost_bps"])
        rows.append({"method": name, **evaluate(r, config["annualization"])})
    for name, evolving, interval in [
        ("Static Base", False, config["update_interval"]),
        ("Static Tool-Calling", False, config["update_interval"]),
        ("EvolveBase", True, 1),
        ("EvolveStrategy", True, config["update_interval"]),
        ("EvolveTrade", True, config["update_interval"]),
    ]:
        r, records, weights = run_agent(prices, config, evolving, interval)
        rows.append({"method": name, **evaluate(r, config["annualization"])})
        if name == "EvolveTrade":
            r.rename("return").to_csv(output / "evolvetrade_daily_returns.csv")
            weights.to_csv(output / "evolvetrade_weights.csv")
            (output / "experience_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for interval in [1, 3, 5, 7]:
        r, _, _ = run_agent(prices, config, True, interval)
        rows.append({"method": f"EvolveTrade-N{interval}", **evaluate(r, config["annualization"])})
    result = pd.DataFrame(rows)
    result.to_csv(output / "metrics.csv", index=False)
    return result


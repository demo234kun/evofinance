import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.deepseek_agent import DeepSeekClient, DeepSeekPolicyAgent, DeepSeekTradingAgent
from evolvetrade.metrics import evaluate

config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
prices = pd.read_csv(ROOT / "data" / "demo" / "prices.csv", index_col=0, parse_dates=True)
prices = prices.iloc[-75:]
client = DeepSeekClient()
trader = DeepSeekTradingAgent(client, config["max_asset_weight"])
updater = DeepSeekPolicyAgent(client)
policy = "Retrieve price, lagged news and code evidence. Cross-check momentum and risk. Allocate long-only across assets and cash."
records, daily_returns, policies = [], [], [policy]
interval = config["update_interval"]
for i in range(60, len(prices) - 1):
    date, next_date = prices.index[i], prices.index[i + 1]
    weights, trace = trader.decide(prices.iloc[: i + 1], policy)
    asset_return = prices.iloc[i + 1] / prices.iloc[i] - 1
    realized = float((weights.drop("CASH") * asset_return).sum())
    record = {
        "decision_date": str(date.date()),
        "feedback_date": str(next_date.date()),
        "policy": policy,
        "weights": {k: float(v) for k, v in weights.items()},
        "trace": trace,
        "realized_return": realized,
    }
    records.append(record)
    daily_returns.append((next_date, realized))
    if len(records) % interval == 0:
        policy = updater.refine(policy, records[-interval:])
        policies.append(policy)
returns = pd.Series(dict(daily_returns)).sort_index()
metrics = evaluate(returns, config["annualization"])
(ROOT / "result" / "deepseek_experience_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
(ROOT / "result" / "deepseek_policy_history.json").write_text(json.dumps(policies, ensure_ascii=False, indent=2), encoding="utf-8")
returns.rename("return").to_csv(ROOT / "result" / "deepseek_daily_returns.csv")
pd.DataFrame([metrics]).to_csv(ROOT / "result" / "deepseek_smoke_metrics.csv", index=False)
print(json.dumps({"decisions": len(records), "policy_versions": len(policies), **metrics}, ensure_ascii=False, indent=2))

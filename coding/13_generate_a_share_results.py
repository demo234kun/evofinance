from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
OUT = RESULT / "a_share_tables"
OUT.mkdir(parents=True, exist_ok=True)
metrics = ["sharpe", "cumulative_return", "max_drawdown", "win_rate", "volatility"]
rule = pd.read_csv(RESULT / "a_share_rule_baselines.csv")
llm = pd.read_csv(RESULT / "a_share_llm_results.csv")
summary = llm.groupby(["method", "period"], as_index=False)[metrics].mean()
main = pd.concat([rule.assign(group="Rule-Based"), summary.assign(group="LLM-Based")], ignore_index=True)
main.to_csv(OUT / "table_A1_main_results.csv", index=False, encoding="utf-8-sig")
stability = llm.groupby(["method", "period"])[["sharpe", "cumulative_return"]].agg(["mean", "std"])
stability.to_csv(OUT / "table_A2_three_seed_stability.csv", encoding="utf-8-sig")
cost = llm.groupby(["method", "period"], as_index=False).agg(turnover=("turnover", "mean"),
    net_sharpe=("net_sharpe", "mean"), net_cumulative_return=("net_cumulative_return", "mean"))
cost.to_csv(OUT / "table_A3_transaction_cost.csv", index=False, encoding="utf-8-sig")

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
TABLES = RESULT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
rule = pd.read_csv(RESULT / "rule_baselines.csv")
llm = pd.read_csv(RESULT / "llm_main_results.csv")
if "protocol" in llm.columns:
    llm = llm[llm["protocol"].eq("protocol_v2")].copy()
metrics = ["sharpe", "cumulative_return", "max_drawdown", "win_rate", "volatility"]
gpt_periods = ["2025-01", "2025-04", "2025-09"]
gemini_periods = ["2025-11", "2026-02", "2026-04"]
main = pd.concat([rule.assign(seed="deterministic"), llm], ignore_index=True)
main.to_csv(TABLES / "table_1_main_results_long.csv", index=False)
for number, periods in [(4, gpt_periods), (5, gemini_periods)]:
    rows = []
    for (method, period), group in llm[llm["period"].isin(periods)].groupby(["method", "period"]):
        rows.append({"method": method, "period": period, "runs": group["seed"].nunique(),
                     "sharpe_mean": group["sharpe"].mean(), "sharpe_std": group["sharpe"].std(ddof=1),
                     "cumulative_return_mean": group["cumulative_return"].mean(),
                     "cumulative_return_std": group["cumulative_return"].std(ddof=1)})
    pd.DataFrame(rows).to_csv(TABLES / f"table_{number}_mean_std.csv", index=False)
cost = llm.groupby(["method", "period"], as_index=False).agg(
    turnover=("turnover", "mean"), net_sharpe=("net_sharpe", "mean"),
    net_cumulative_return=("net_cumulative_return", "mean"))
cost.to_csv(TABLES / "table_6_transaction_cost.csv", index=False)
cash = llm.groupby(["method", "period"], as_index=False)["average_cash"].mean()
cash_rows = []
for period in gpt_periods + gemini_periods:
    baseline = "Static Tool-Calling" if period in gpt_periods else "Static Base"
    base = cash[(cash["period"] == period) & (cash["method"] == baseline)]
    evolve = cash[(cash["period"] == period) & (cash["method"] == "EvolveTrade")]
    if not base.empty and not evolve.empty:
        b, e = float(base.iloc[0]["average_cash"]), float(evolve.iloc[0]["average_cash"])
        cash_rows.append({"period": period, "baseline": baseline, "baseline_cash": b,
                          "evolvetrade_cash": e, "delta_cash": e - b})
pd.DataFrame(cash_rows).to_csv(TABLES / "table_7_cash_allocation.csv", index=False)
if (RESULT / "interval_sensitivity.csv").exists():
    interval = pd.read_csv(RESULT / "interval_sensitivity.csv")
    interval.groupby("update_interval")[metrics].mean().to_csv(TABLES / "table_3_update_interval.csv")
if (RESULT / "long_horizon.csv").exists():
    long = pd.read_csv(RESULT / "long_horizon.csv")
    long.groupby("method")[metrics].agg(["mean", "std"]).to_csv(TABLES / "table_2_long_horizon.csv")
for path in TABLES.glob("*.csv"):
    if path.stat().st_size:
        try:
            pd.read_csv(path).to_html(path.with_suffix(".html"), index=False, border=0, float_format=lambda x: f"{x:.4f}")
        except pd.errors.EmptyDataError:
            pass


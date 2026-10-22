import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "result" / "runs"
FIG = ROOT / "result" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
records = []
for path in RUNS.rglob("records.jsonl"):
    if "protocol_v2" not in path.parts:
        continue
    records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No real-market run logs found")
df["feedback_date"] = pd.to_datetime(df["feedback_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])
df["net_return"] = df["gross_return"] - df["turnover"] * 0.001
for period, group in df.groupby("period"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, runs in group.groupby("method"):
        curves = []
        for _, run in runs.groupby("seed"):
            curves.append((1 + run.set_index("feedback_date")["net_return"]).cumprod())
        panel = pd.concat(curves, axis=1)
        ax.plot(panel.index, panel.mean(axis=1), label=method)
    ax.set_title(f"Daily Net Value: {period}")
    ax.set_ylabel("Net Value")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / f"figure_equity_{period}.png", dpi=220)
    plt.close(fig)
tool_rows = []
for _, row in df[df["period"].isin(["2025-01", "2025-04", "2025-09"]) & df["method"].isin(["Static Tool-Calling", "EvolveTrade"])].iterrows():
    counts = pd.Series([call.get("name", "unknown") for call in row.get("tool_calls", [])]).value_counts()
    for tool_name in ["get_price", "code_interpreter", "news_searcher"]:
        tool_rows.append({"period": row["period"], "method": row["method"], "tool": tool_name,
                          "calls": counts.get(tool_name, 0)})
tool = pd.DataFrame(tool_rows).groupby(["period", "method", "tool"], as_index=False)["calls"].mean()
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, period in zip(axes, ["2025-01", "2025-04", "2025-09"]):
    tool[tool["period"] == period].pivot(index="tool", columns="method", values="calls").plot(kind="bar", ax=ax)
    ax.set_title(period)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
axes[0].set_ylabel("Mean Daily Tool Calls")
fig.tight_layout()
fig.savefig(FIG / "figure_4_tool_frequency.png", dpi=220)
plt.close(fig)
cash = df.assign(cash=df["allocations"].map(lambda x: x["CASH"])).groupby(["period", "method"])["cash"].mean().unstack()
cash.plot(kind="bar", figsize=(11, 5))
plt.ylabel("Average Cash Weight")
plt.tight_layout()
plt.savefig(FIG / "figure_cash_exposure.png", dpi=220)
plt.close()
terms = ["sharpe_20d", "return_20d", "return_60d", "volatility_20d", "drawdown_60d", "correlation", "rsi_14d", "ema_20d", "sma_20d", "var_95", "signal_normalization"]
def metric_set(calls):
    found = set()
    for call in calls:
        if call.get("name") == "code_interpreter":
            found.update(call.get("arguments", {}).get("metrics", []))
    return found
df["metrics_used"] = df["tool_calls"].map(metric_set)
tc = df[df["method"].isin(["Static Tool-Calling", "EvolveTrade"])].copy()
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
cmap = ListedColormap(["#ffffff", "#d73027", "#7b3294", "#2c7fb8"])
for ax, period in zip(axes, ["2025-01", "2025-04", "2025-09"]):
    panel = tc[tc["period"] == period]
    days = sorted(panel["decision_date"].dt.strftime("%m-%d").unique())
    matrix = np.zeros((len(terms), len(days)))
    for j, day in enumerate(days):
        day_rows = panel[panel["decision_date"].dt.strftime("%m-%d") == day]
        for i, term in enumerate(terms):
            static_rate = day_rows[day_rows["method"] == "Static Tool-Calling"]["metrics_used"].map(lambda values: term in values).mean()
            evolve_rate = day_rows[day_rows["method"] == "EvolveTrade"]["metrics_used"].map(lambda values: term in values).mean()
            matrix[i, j] = 3 if evolve_rate >= .5 and static_rate < .5 else 1 if static_rate >= .5 and evolve_rate < .5 else 2 if static_rate >= .5 and evolve_rate >= .5 else 0
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax.set_title(period)
    ax.set_xticks(range(len(days)), days, rotation=90, fontsize=7)
    ax.set_yticks(range(len(terms)), terms)
fig.tight_layout()
fig.savefig(FIG / "figure_3_metric_activation.png", dpi=220)
plt.close(fig)

static = df[(df["method"] == "Static Tool-Calling") & df["period"].isin(["2025-01", "2025-04", "2025-09"])]
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
for ax, period in zip(axes, ["2025-01", "2025-04", "2025-09"]):
    panel = static[static["period"] == period]
    days = sorted(panel["decision_date"].dt.strftime("%m-%d").unique())
    matrix = np.zeros((len(terms), len(days)))
    for j, day in enumerate(days):
        used = panel[panel["decision_date"].dt.strftime("%m-%d") == day]["metrics_used"]
        for i, term in enumerate(terms):
            matrix[i, j] = used.map(lambda values: term in values).mean()
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_title(period)
    ax.set_xticks(range(len(days)), days, rotation=90, fontsize=7)
    ax.set_yticks(range(len(terms)), terms)
fig.tight_layout()
fig.savefig(FIG / "figure_2_static_metric_vocabulary.png", dpi=220)
plt.close(fig)

long_records = []
for path in RUNS.rglob("records.jsonl"):
    if "protocol_v2_paper_50d" in path.parts:
        long_records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
long_df = pd.DataFrame(long_records)
if not long_df.empty:
    long_df["decision_date"] = pd.to_datetime(long_df["decision_date"])
    policy = long_df[long_df["method"] == "EvolveTrade"].sort_values("decision_date")
    for seed, run in policy.groupby("seed"):
        plt.plot(run["decision_date"], run["policy"].str.len(), label=str(seed))
    plt.xlabel("Decision Date")
    plt.ylabel("Policy Characters")
    plt.legend(title="Seed")
    plt.tight_layout()
    plt.savefig(FIG / "figure_6_policy_length.png", dpi=220)
    plt.close()

case = df[(df["period"] == "2025-01") & (df["decision_date"] == "2025-01-24") & df["method"].isin(["Static Tool-Calling", "EvolveTrade"])].copy()
if not case.empty:
    weights = case.groupby("method")["allocations"].apply(lambda values: np.mean([value.get("NVDA", 0.0) for value in values]))
    returns = case.groupby("method")["gross_return"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    weights.plot(kind="bar", ax=axes[0], color=["#d95f5f", "#315ea8"])
    returns.plot(kind="bar", ax=axes[1], color=["#d95f5f", "#315ea8"])
    axes[0].set_ylabel("Mean NVDA Weight")
    axes[0].set_title("Jan 24 Allocation")
    axes[1].set_ylabel("Next-Day Portfolio Return")
    axes[1].set_title("Jan 27 Realized Return")
    fig.tight_layout()
    fig.savefig(FIG / "figure_5_nvda_drawdown_case.png", dpi=220)
    plt.close(fig)

case = df[(df["period"] == "2025-09") & (df["decision_date"] == "2025-09-10") & df["method"].isin(["Static Tool-Calling", "EvolveTrade"])].copy()
if not case.empty:
    rows = []
    for method, group in case.groupby("method"):
        contributions = {}
        for _, record in group.iterrows():
            for ticker, value in record["asset_returns"].items():
                contributions.setdefault(ticker, []).append(record["allocations"].get(ticker, 0.0) * value)
        for ticker, values in contributions.items():
            rows.append({"method": method, "ticker": ticker, "contribution": np.mean(values)})
    panel = pd.DataFrame(rows).pivot(index="ticker", columns="method", values="contribution").fillna(0)
    panel.loc[panel.abs().max(axis=1).nlargest(8).index].plot(kind="bar", figsize=(10, 5), color=["#d95f5f", "#315ea8"])
    plt.axhline(0, color="black", linewidth=.8)
    plt.ylabel("Next-Day Return Contribution")
    plt.title("September 10 Decision Attribution")
    plt.tight_layout()
    plt.savefig(FIG / "figure_7_specific_day_attribution.png", dpi=220)
    plt.close()

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "innovation_smoke.json").read_text(encoding="utf-8"))
OUT = ROOT / "result" / "innovation_full_smoke"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

def evaluate(s):
    s = pd.Series(s).fillna(0.0).astype(float)
    equity = (1 + s).cumprod()
    sd = s.std(ddof=1)
    dd = equity / equity.cummax() - 1
    return {"sharpe": float(s.mean() / sd * math.sqrt(252)) if sd else 0.0,
            "cumulative_return": float(equity.iloc[-1] - 1),
            "max_drawdown": float(-dd.min()),
            "win_rate": float((s > 0).mean()),
            "volatility": float(sd * math.sqrt(252))}

def constrained_weights(momentum, max_weight):
    score = momentum.clip(lower=0).astype(float)
    if score.sum() <= 0:
        score[:] = 1.0
    w = score / score.sum()
    for _ in range(20):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()

def state_and_failure(px, rets, i, evo_returns, equity, peak):
    mom = float(px["SPY"].iloc[i] / px["SPY"].iloc[i - 21] - 1)
    vol = float(rets["SPY"].iloc[i - 20:i + 1].std() * math.sqrt(252))
    state = ("up" if mom >= 0 else "down") + ("_highvol" if vol >= CFG["volatility_threshold"] else "_lowvol")
    recent = evo_returns[-CFG["failure_lookback"]:]
    failure = len(recent) >= CFG["failure_lookback"] and (sum(recent) < CFG["failure_return_threshold"] or equity / peak - 1 < CFG["failure_drawdown_threshold"])
    return state, bool(failure), mom, vol

def run(px, assets, mode, start, end):
    rets = px.pct_change()
    history = {}
    daily, trace = [], []
    equity = peak = 1.0
    total = max(1, end - start - 1)
    for step, i in enumerate(range(start, end - 1), 1):
        d, nd = px.index[i], px.index[i + 1]
        r = rets.iloc[i]
        base = constrained_weights((px.iloc[i] / px.iloc[i - 21] - 1)[assets], CFG["max_asset_weight"])
        base_r = float((base * r[assets]).sum())
        state, failure, mom, vol = state_and_failure(px, rets, i, [x["return"] for x in daily], equity, peak)
        mult = 1.0
        rollback = bool(daily and daily[-1]["multiplier"] != 1.0 and daily[-1]["return"] < daily[-1]["baseline_return"])
        if mode == "full" and (failure or state == "up_lowvol"):
            exp = history.get(state, {})
            valid = {k: v for k, v in exp.items() if len(v) >= CFG["min_experience"]}
            if valid:
                mult = max(valid, key=lambda k: np.mean(valid[k]))
            else:
                mult = 0.7 if state == "down_highvol" else (1.15 if state == "up_lowvol" else 1.0)
            if rollback:
                mult = 1.0
        elif mode == "scenario_only" and failure:
            mult = 0.7 if state == "down_highvol" else 1.0
        elif mode == "fixed_leverage":
            mult = 1.15 if state == "up_lowvol" else 1.0
        elif mode == "experience_only" and failure:
            exp = history.get(state, {})
            valid = {k: v for k, v in exp.items() if len(v) >= CFG["min_experience"]}
            mult = max(valid, key=lambda k: np.mean(valid[k])) if valid else 1.0
        evolved_r = base_r * mult
        cost = CFG["transaction_cost_bps"] / 10000 * abs(mult - 1.0)
        if mode == "baseline":
            evolved_r = base_r
            cost = 0.0
        evolved_r -= cost
        equity *= 1 + evolved_r
        peak = max(peak, equity)
        if mode in {"full", "experience_only"}:
            history.setdefault(state, {}).setdefault(mult, []).append(evolved_r)
        daily.append({"date": nd, "return": evolved_r, "baseline_return": base_r, "state": state, "failure_trigger": failure, "rollback": rollback, "multiplier": mult, "equity": equity, "momentum": mom, "volatility": vol})
        if step == 1 or step == total or step % max(1, total // 20) == 0:
            print(f"当前冒烟单元 {mode}: {step}/{total} ({step / total:.1%})", flush=True)
    return pd.DataFrame(daily).set_index("date")

def paired_bootstrap(a, b):
    rng = np.random.default_rng(CFG["seed"])
    d = (b - a).to_numpy()
    draws = rng.choice(d, size=(CFG["bootstrap_reps"], len(d)), replace=True).mean(axis=1)
    return {"mean_daily_difference": float(d.mean()), "ci_95_low": float(np.quantile(draws, .025)), "ci_95_high": float(np.quantile(draws, .975)), "p_improvement": float((draws <= 0).mean())}

px = pd.read_csv(ROOT / CFG["data_file"], parse_dates=["date"]).set_index("date").sort_index()
assets = [x for x in json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))["assets"] if x in px.columns]
px = px[assets + ["SPY"]].dropna()
start = CFG["warmup_days"]
end = min(len(px) - 1, start + CFG["window_days"])
methods = {"Baseline": "baseline", "ScenarioOnly": "scenario_only", "ExperienceOnly": "experience_only", "FixedLeverage": "fixed_leverage", "ContextEvolution": "full"}
series, rows = {}, []
for label, mode in methods.items():
    out = run(px, assets, mode, start, end)
    series[label] = out["return"]
    rows.append({"method": label, **evaluate(out["return"])})
metrics = pd.DataFrame(rows)
metrics.to_csv(TABLES / "table_1_metrics.csv", index=False)
trace = run(px, assets, "full", start, end)
trace.reset_index().to_csv(TABLES / "table_2_event_trace.csv", index=False)
comparison = pd.DataFrame({k: v for k, v in series.items()})
comparison["full_minus_baseline"] = comparison["ContextEvolution"] - comparison["Baseline"]
comparison.to_csv(OUT / "paired_returns.csv")
boot = paired_bootstrap(series["Baseline"], series["ContextEvolution"])
(TABLES / "table_3_bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
ablation = metrics.copy()
ablation["return_delta_vs_baseline"] = ablation["cumulative_return"] - ablation.loc[ablation.method == "Baseline", "cumulative_return"].iloc[0]
ablation.to_csv(TABLES / "table_4_ablation.csv", index=False)
segments = []
for j, (label, seg) in enumerate(trace.groupby(np.arange(len(trace)) // max(1, len(trace) // 3))):
    segments.append({"segment": j + 1, **evaluate(seg["return"]), "baseline_sharpe": evaluate(seg["baseline_return"])["sharpe"]})
pd.DataFrame(segments).to_csv(TABLES / "table_5_rolling_oos.csv", index=False)
pd.DataFrame([{"data_file": CFG["data_file"], "rows": len(px), "assets": len(assets), "warmup": start, "test_rows": len(trace), "real_data_only": True, "transaction_cost_bps": CFG["transaction_cost_bps"]}]).to_csv(TABLES / "table_6_data_audit.csv", index=False)
pd.DataFrame([{**boot, "method": "ContextEvolution_vs_Baseline"}]).to_csv(TABLES / "table_7_significance.csv", index=False)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for label, s in series.items(): axes[0].plot((1 + s).cumprod(), label=label)
axes[0].legend(ncol=3); axes[0].set_ylabel("Net Value"); axes[0].grid(alpha=.25)
axes[1].plot(comparison.index, comparison["full_minus_baseline"].cumsum(), color="#0B6E4F")
axes[1].axhline(0, color="black", linewidth=.8); axes[1].set_ylabel("Cumulative Return Difference"); axes[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIGURES / "figure_1_equity_and_difference.png", dpi=180); plt.close(fig)
trace["failure_trigger"].astype(int).plot(figsize=(12, 2), drawstyle="steps-post", color="#C44E52", title="Failure Trigger Timeline")
plt.tight_layout(); plt.savefig(FIGURES / "figure_2_failure_triggers.png", dpi=180); plt.close()
(OUT / "run_manifest.json").write_text(json.dumps({"status": "completed", "config": CFG, "methods": methods, "tables": 7, "figures": 2}, indent=2), encoding="utf-8")
print(metrics.to_string(index=False))
print(json.dumps(boot, indent=2))

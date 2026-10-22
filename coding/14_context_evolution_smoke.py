import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result" / "innovation_smoke"
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
px = pd.read_csv(ROOT / "data" / "real" / "us_daily_adjusted_close.csv", parse_dates=["date"]).set_index("date").sort_index()
assets = [x for x in cfg["assets"] if x in px.columns]
px = px[assets + ["SPY"]].dropna(how="all")
rets = px.pct_change()

def metrics(s):
    s = s.fillna(0.0).astype(float)
    eq = (1 + s).cumprod()
    sd = s.std(ddof=1)
    dd = eq / eq.cummax() - 1
    return {"sharpe": float(s.mean() / sd * math.sqrt(252)) if sd else 0.0,
            "cumulative_return": float(eq.iloc[-1] - 1),
            "max_drawdown": float(-dd.min()),
            "win_rate": float((s > 0).mean()),
            "volatility": float(sd * math.sqrt(252))}

def base_weights(momentum):
    score = momentum.clip(lower=0)
    if score.sum() <= 0:
        score = pd.Series(1.0, index=momentum.index)
    w = score / score.sum()
    return w.clip(upper=0.35).pipe(lambda x: x / x.sum())

history = {}
base_daily, evo_daily, rows = [], [], []
equity_b = equity_e = 1.0
for i in range(61, len(px) - 1):
    d, nd = px.index[i], px.index[i + 1]
    r = rets.iloc[i]
    m = px.iloc[i] / px.iloc[i - 21] - 1
    w = base_weights(m[assets])
    br = float((w * r[assets]).sum())
    market_mom = float(px["SPY"].iloc[i] / px["SPY"].iloc[i - 21] - 1)
    vol = float(rets["SPY"].iloc[max(1, i - 20):i + 1].std() * math.sqrt(252))
    state = ("up" if market_mom >= 0 else "down") + ("_highvol" if vol >= 0.22 else "_lowvol")
    recent = pd.Series([x[1] for x in evo_daily[-5:]]) if evo_daily else pd.Series(dtype=float)
    peak = max([x["equity"] for x in rows[-20:]] or [1.0])
    failure = len(recent) >= 5 and (recent.sum() < 0 or equity_e / peak - 1 < -0.08)
    candidates = [0.7, 1.0, 1.15]
    if state in history and len(history[state]) >= 5:
        means = {a: np.mean(history[state].get(a, [0.0])) for a in candidates}
        mult = max(candidates, key=lambda a: means[a])
    else:
        mult = 0.7 if state == "down_highvol" else (1.15 if state == "up_lowvol" else 1.0)
    if not failure:
        mult = 1.0
    ew = w * mult
    evo_r = float((ew * r[assets]).sum())
    cost = 0.0005 * float((ew - w).abs().sum())
    evo_r -= cost
    equity_b *= 1 + br
    equity_e *= 1 + evo_r
    history.setdefault(state, {}).setdefault(mult, []).append(evo_r)
    base_daily.append((nd, br)); evo_daily.append((nd, evo_r))
    rows.append({"date": str(nd.date()), "state": state, "failure_trigger": bool(failure), "multiplier": mult, "baseline_return": br, "evolved_return": evo_r, "equity": equity_e})

b = pd.Series(dict(base_daily)).sort_index()
e = pd.Series(dict(evo_daily)).sort_index()
summary = pd.DataFrame([{**{"method": "Baseline"}, **metrics(b)}, {**{"method": "Context-Evolution"}, **metrics(e)}])
summary.to_csv(OUT / "metrics.csv", index=False)
delta = {f"delta_{c}": float(summary.loc[1, c] - summary.loc[0, c]) for c in ["sharpe", "cumulative_return", "max_drawdown", "win_rate", "volatility"]}
(OUT / "comparison.json").write_text(json.dumps({"data": "us_daily_adjusted_close.csv", "real_data_only": True, **delta, "improved_return": delta["delta_cumulative_return"] > 0, "improved_sharpe": delta["delta_sharpe"] > 0, "improved_drawdown": delta["delta_max_drawdown"] < 0}, ensure_ascii=False, indent=2), encoding="utf-8")
pd.DataFrame(rows).to_csv(OUT / "daily_trace.csv", index=False)
paired = pd.DataFrame({"baseline": b, "context_evolution": e})
paired["difference"] = paired["context_evolution"] - paired["baseline"]
paired.to_csv(OUT / "paired_returns.csv")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot((1 + b).cumprod(), label="Baseline", color="#777777")
ax.plot((1 + e).cumprod(), label="Context-Evolution", color="#0B6E4F")
tr = pd.DataFrame(rows)
for d in tr.loc[tr.failure_trigger, "date"]:
    ax.axvline(pd.Timestamp(d), color="#C44E52", alpha=0.18, linewidth=0.8)
ax.set_title("Innovation Smoke Test: Real Market Data")
ax.set_ylabel("Cumulative Net Value")
ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "equity_comparison.png", dpi=180); plt.close(fig)
print(summary.to_string(index=False))

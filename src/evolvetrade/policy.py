from dataclasses import asdict, dataclass, replace
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    momentum_weight: float = 0.65
    reversal_weight: float = 0.15
    risk_weight: float = 0.20
    cash_floor: float = 0.05
    temperature: float = 0.75
    version: int = 0

    def text(self) -> str:
        return (
            f"Policy v{self.version}: retrieve prices and lagged news; compute momentum, reversal and risk; "
            f"weights=({self.momentum_weight:.3f},{self.reversal_weight:.3f},{self.risk_weight:.3f}); "
            f"cash_floor={self.cash_floor:.3f}; cross-check signals before allocation."
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"text": self.text()}


class PolicyAgent:
    def refine(self, policy: Policy, batch_returns: pd.Series, trace: list[dict]) -> Policy:
        mean = float(batch_returns.mean())
        vol = float(batch_returns.std(ddof=0))
        equity = (1 + batch_returns).cumprod()
        mdd = float(-(equity / equity.cummax() - 1).min())
        if mean < 0 or mdd > 0.035:
            return replace(
                policy,
                momentum_weight=max(0.35, policy.momentum_weight - 0.08),
                reversal_weight=min(0.35, policy.reversal_weight + 0.03),
                risk_weight=min(0.40, policy.risk_weight + 0.05),
                cash_floor=min(0.35, policy.cash_floor + 0.05),
                temperature=max(0.45, policy.temperature - 0.05),
                version=policy.version + 1,
            )
        if mean > 0 and vol < 0.02:
            return replace(
                policy,
                momentum_weight=min(0.80, policy.momentum_weight + 0.04),
                reversal_weight=max(0.05, policy.reversal_weight - 0.02),
                cash_floor=max(0.02, policy.cash_floor - 0.02),
                version=policy.version + 1,
            )
        return replace(policy, version=policy.version + 1)


def allocations(history: pd.DataFrame, policy: Policy, max_weight: float) -> tuple[pd.Series, dict]:
    r = history.pct_change()
    mom20 = history.iloc[-1] / history.iloc[-21] - 1 if len(history) >= 21 else r.mean()
    mom60 = history.iloc[-1] / history.iloc[-61] - 1 if len(history) >= 61 else mom20
    reversal = -r.tail(5).mean()
    risk = r.tail(min(60, len(r))).std().replace(0, np.nan).fillna(risk_default(r))
    z = lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    score = policy.momentum_weight * z(0.5 * mom20 + 0.5 * mom60) + policy.reversal_weight * z(reversal) - policy.risk_weight * z(risk)
    raw = np.exp((score - score.max()) / max(policy.temperature, 1e-6))
    weights = pd.Series(raw / raw.sum() * (1 - policy.cash_floor), index=history.columns)
    for _ in range(20):
        excess = (weights - max_weight).clip(lower=0).sum()
        weights = weights.clip(upper=max_weight)
        room = (max_weight - weights).clip(lower=0)
        if excess < 1e-12 or room.sum() < 1e-12:
            break
        weights += excess * room / room.sum()
    weights["CASH"] = max(0.0, 1.0 - weights.sum())
    trace = {
        "policy_version": policy.version,
        "tool_calls": ["get_price", "news_searcher", "code_interpreter"],
        "metrics": ["return_20d", "return_60d", "volatility", "drawdown_proxy"],
        "rationale": "momentum-reversal-risk synthesis under current policy",
    }
    return weights, trace


def risk_default(returns: pd.DataFrame) -> float:
    value = float(returns.stack().std())
    return value if np.isfinite(value) and value > 0 else 0.01

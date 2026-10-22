import json
from pathlib import Path
import random
import re
import numpy as np
import pandas as pd

from .deepseek_agent import DeepSeekClient
from .metrics import evaluate
from .real_data import COMPANIES


INITIAL_STRATEGY = """Use 20-day and 60-day momentum as primary signals. Use point-in-time news as confirmation and annualized volatility to scale weights. Cap a stock at 35%. Hold 5%-10% cash normally and 15%-20% when broad risk is negative. When momentum and news conflict, halve the intended exposure. Minimize turnover unless evidence materially changes."""

TOOL_PROTOCOL = """For every decision retrieve all asset prices with a 3-12 month lookback, run quantitative analysis, and search news available only through t-1. Compute at least 20-day return, annualized volatility and Sharpe; optionally use 60-day return, drawdown, correlation, RSI, EMA, SMA, VaR and signal normalization. Cite price, computed metrics, news and sizing logic in each asset trace."""

TOOLS = [
    {"type": "function", "function": {"name": "get_price", "description": "Retrieve point-in-time daily closes through the decision date.", "parameters": {"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}, "lookback_days": {"type": "integer", "minimum": 60, "maximum": 365}}, "required": ["tickers", "lookback_days"]}}},
    {"type": "function", "function": {"name": "code_interpreter", "description": "Compute requested quantitative metrics on point-in-time prices.", "parameters": {"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}, "metrics": {"type": "array", "items": {"type": "string", "enum": ["return_5d", "return_20d", "return_60d", "return_120d", "volatility_20d", "volatility_60d", "sharpe_20d", "sharpe_60d", "drawdown_60d", "correlation", "rsi_14d", "ema_20d", "sma_20d", "var_95", "signal_normalization"]}}}, "required": ["tickers", "metrics"]}}},
    {"type": "function", "function": {"name": "news_searcher", "description": "Search point-in-time news through t-1 only.", "parameters": {"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}, "lookback_days": {"type": "integer", "minimum": 1, "maximum": 30}}, "required": ["tickers", "lookback_days"]}}},
]


def load_real_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(root / "data" / "real" / "us_daily_adjusted_close.csv", index_col=0, parse_dates=True).sort_index()
    news = pd.read_parquet(root / "data" / "real" / "news" / "newsmood_news.parquet")
    google_path = root / "data" / "real" / "news" / "google_news_historical.parquet"
    if google_path.exists():
        google = pd.read_parquet(google_path)
        common = [column for column in news.columns if column in google.columns]
        news = pd.concat([news[common], google[common]], ignore_index=True)
    news["ts"] = pd.to_datetime(news["ts"], utc=True).dt.tz_convert(None)
    return prices, news


def metric_snapshot(history: pd.DataFrame) -> dict:
    ret = history.pct_change()
    last = history.iloc[-1]
    def vals(series):
        return {k: None if not np.isfinite(v) else round(float(v), 6) for k, v in series.items()}
    return {
        "close": vals(last),
        "return_5d": vals(last / history.iloc[-6] - 1),
        "return_20d": vals(last / history.iloc[-21] - 1),
        "return_60d": vals(last / history.iloc[-61] - 1),
        "annualized_volatility_20d": vals(ret.tail(20).std() * np.sqrt(252)),
        "sharpe_20d": vals(ret.tail(20).mean() / (ret.tail(20).std() + 1e-12) * np.sqrt(252)),
        "drawdown_60d": vals(last / history.tail(60).max() - 1),
    }


def news_snapshot(news: pd.DataFrame, tickers: list[str], date: pd.Timestamp) -> dict:
    cutoff = date.normalize() - pd.Timedelta(days=1)
    start = cutoff - pd.Timedelta(days=4)
    window = news[(news["ts"] >= start) & (news["ts"] < cutoff + pd.Timedelta(days=1))]
    result = {}
    text = (window["title"].fillna("") + " " + window["description"].fillna(""))
    for ticker in tickers:
        terms = [ticker, COMPANIES.get(ticker, ticker)]
        mask = text.str.contains("|".join(re.escape(term) for term in terms), case=False, regex=True, na=False)
        rows = window.loc[mask].sort_values("ts", ascending=False).head(3)
        result[ticker] = [{"ts": str(r.ts), "title": str(r.title), "source": str(r.source)} for r in rows.itertuples()]
    return result


def normalize_weights(raw: dict, tickers: list[str], cap: float = 0.35) -> dict:
    weights = {t: max(0.0, min(cap, float(raw.get(t, 0.0)))) for t in tickers}
    cash = max(0.0, float(raw.get("CASH", 0.0)))
    target_equity = min(1.0 - min(cash, 1.0), sum(weights.values()))
    if sum(weights.values()) > 0:
        scale = target_equity / sum(weights.values())
        weights = {k: v * scale for k, v in weights.items()}
    weights["CASH"] = 1.0 - sum(weights.values())
    return weights


def quantitative_tool(history: pd.DataFrame, tickers: list[str], metrics: list[str]) -> dict:
    cols = [ticker for ticker in tickers if ticker in history.columns]
    frame = history[cols]
    ret = frame.pct_change()
    last = frame.iloc[-1]
    output = {}
    for metric in metrics:
        if metric.startswith("return_"):
            window = int(metric.split("_")[1][:-1])
            value = last / frame.iloc[-min(window + 1, len(frame))] - 1
        elif metric.startswith("volatility_"):
            window = int(metric.split("_")[1][:-1])
            value = ret.tail(window).std() * np.sqrt(252)
        elif metric.startswith("sharpe_"):
            window = int(metric.split("_")[1][:-1])
            sample = ret.tail(window)
            value = sample.mean() / (sample.std() + 1e-12) * np.sqrt(252)
        elif metric == "drawdown_60d":
            value = last / frame.tail(60).max() - 1
        elif metric == "rsi_14d":
            delta = frame.diff().tail(14)
            value = 100 - 100 / (1 + delta.clip(lower=0).mean() / (-delta.clip(upper=0).mean() + 1e-12))
        elif metric == "ema_20d":
            value = frame.ewm(span=20, adjust=False).mean().iloc[-1]
        elif metric == "sma_20d":
            value = frame.tail(20).mean()
        elif metric == "var_95":
            value = ret.tail(60).quantile(0.05)
        elif metric == "signal_normalization":
            value = (last / frame.iloc[-min(21, len(frame))] - 1)
            value = (value - value.mean()) / (value.std() + 1e-12)
        elif metric == "correlation":
            output[metric] = ret.tail(60).corr().round(4).to_dict()
            continue
        else:
            continue
        output[metric] = {k: None if not np.isfinite(v) else round(float(v), 6) for k, v in value.items()}
    return output


def run_tool_loop(client: DeepSeekClient, system: str, user: str, history: pd.DataFrame, news: pd.DataFrame, date: pd.Timestamp, tickers: list[str], checkpoint_path: Path | None = None) -> tuple[dict, list[dict]]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    trace = []
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("system") == system and saved.get("user") == user:
            messages, trace = saved["messages"], saved["trace"]
            if "response" in saved:
                return saved["response"], trace
    required = {"get_price", "code_interpreter", "news_searcher"}
    while True:
        ready = required.issubset({item["name"] for item in trace})
        if ready and messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": "All required tools have returned evidence. Use this evidence to return the final allocation JSON now."})
        message = client.tool_turn(messages, TOOLS, tool_choice="none" if ready else "auto")
        calls = message.get("tool_calls") or []
        if not calls:
            content = re.sub(r"^```(?:json)?|```$", "", (message.get("content") or "").strip(), flags=re.MULTILINE).strip()
            start = content.find("{")
            if start >= 0:
                try:
                    result, _ = json.JSONDecoder().raw_decode(content[start:])
                    if not isinstance(result, dict) or not isinstance(result.get("allocations"), dict) or not result["allocations"]:
                        raise json.JSONDecodeError("Missing allocations", content, 0)
                    if not ready:
                        messages.append(message)
                        messages.append({"role": "user", "content": "Retrieve the missing required tools before allocating: " + ", ".join(sorted(required - {item["name"] for item in trace}))})
                        continue
                    if checkpoint_path is not None:
                        temporary = checkpoint_path.with_suffix(".tmp")
                        temporary.write_text(json.dumps({"system": system, "user": user, "messages": messages, "trace": trace, "response": result}, ensure_ascii=False), encoding="utf-8")
                        temporary.replace(checkpoint_path)
                    return result, trace
                except json.JSONDecodeError:
                    pass
            messages.append({"role": "user", "content": "Return the final allocation now as one valid JSON object with traceability and allocations."})
            continue
        messages.append(message)
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"].get("arguments") or "{}")
            requested = [ticker for ticker in args.get("tickers", tickers) if ticker in tickers]
            if name == "get_price":
                days = min(365, max(60, int(args.get("lookback_days", 180))))
                result = history[requested].tail(days).round(4).to_dict(orient="split")
                result["index"] = [str(pd.Timestamp(x).date()) for x in result["index"]]
            elif name == "code_interpreter":
                result = quantitative_tool(history, requested, args.get("metrics", ["return_20d", "volatility_20d", "sharpe_20d"]))
            elif name == "news_searcher":
                result = news_snapshot(news, requested, date)
            else:
                result = {"error": "unknown tool"}
            trace.append({"name": name, "arguments": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
        if checkpoint_path is not None:
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"system": system, "user": user, "messages": messages, "trace": trace}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(checkpoint_path)


def trading_prompt(method: str, policy: str, date: pd.Timestamp, previous: dict, prices: dict, news: dict) -> tuple[str, str]:
    tool_free = method in {"Static Base", "EvolveBase"}
    system = "You are a professional portfolio manager in a daily-close environment. " + policy
    if not tool_free:
        system += "\n" + TOOL_PROTOCOL
    payload = {
        "decision_date": str(date.date()),
        "previous_portfolio": previous,
        "available_assets": list(prices["close"]) + ["CASH"],
        "constraints": {"long_only": True, "sum": 1.0, "max_stock_weight": 0.35, "news_cutoff": "t-1"},
        "output_schema": {"traceability": {"TICKER": {"reasoning": "string", "metrics_used": ["string"]}}, "allocations": {"TICKER": 0.0, "CASH": 0.0}},
    }
    if tool_free:
        payload["price_and_code_context"] = prices
        payload["prefetched_news_context"] = news
    else:
        payload["instruction"] = "Use the tools yourself. Call all three tools before returning the allocation JSON."
    return system, json.dumps(payload, ensure_ascii=False)


def update_policy(client: DeepSeekClient, method: str, policy: str, batch: list[dict]) -> tuple[str, dict]:
    scope = "high-level trading strategy only; preserve the fixed tool protocol" if method == "EvolveStrategy" else "the complete operational policy, including tool use, evidence verification, risk control and allocation"
    result = client.complete(
        "You are a System Prompt Updater for a daily-close portfolio agent. Analyze allocations, per-asset rationale, actual per-asset returns, tool traces, and portfolio returns. Link concrete evidence to specific policy lines, preserve effective rules, make targeted edits, and avoid one-day overfitting. Return JSON only.",
        json.dumps({"update_scope": scope, "current_policy": policy, "completed_experience_batch": batch, "output_schema": {"reasoning": {"observations": [], "causal_policy_lines": [], "changes": [], "overfit_check": "string"}, "policy": "full updated policy"}}, ensure_ascii=False),
    )
    return str(result.get("policy", policy)), result


def run_llm_method(root: Path, config: dict, method: str, period: str, seed: int, update_interval: int = 5, run_tag: str | None = None) -> dict:
    prices, news = load_real_data(root)
    start, end = map(pd.Timestamp, config["evaluation_windows"][period])
    decision_dates = prices.loc[start:end].index
    tickers = config["assets"]
    out = root / "result" / "runs" / period / method.replace(" ", "_")
    if run_tag:
        out = out / run_tag
    out = out / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "records.jsonl"
    existing = []
    if log_path.exists():
        existing = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {r["decision_date"] for r in existing}
    random.seed(seed)
    np.random.seed(seed)
    client = DeepSeekClient()
    policy = existing[-1]["next_policy"] if existing else INITIAL_STRATEGY
    previous = existing[-1]["allocations"] if existing else {**{t: 0.0 for t in tickers}, "CASH": 1.0}
    records = list(existing)
    for date in decision_dates[:-1]:
        if str(date.date()) in completed:
            continue
        history = prices.loc[:date, tickers].tail(253)
        metrics = metric_snapshot(history)
        headlines = news_snapshot(news, tickers, date)
        system, user = trading_prompt(method, policy, date, previous, metrics, headlines)
        if method in {"Static Base", "EvolveBase"}:
            response = client.complete(system, user)
            tool_trace = []
        else:
            response, tool_trace = run_tool_loop(client, system, user, history, news, date, tickers, out / f"tool_checkpoint_{date.date()}.json")
        weights = normalize_weights(response.get("allocations", {}), tickers)
        next_date = prices.index[prices.index.get_loc(date) + 1]
        asset_returns = prices.loc[next_date, tickers] / prices.loc[date, tickers] - 1
        gross = float(sum(weights[t] * asset_returns[t] for t in tickers))
        turnover = 0.5 * sum(abs(weights[k] - previous.get(k, 0.0)) for k in weights)
        record = {
            "method": method, "period": period, "seed": seed,
            "decision_date": str(date.date()), "feedback_date": str(next_date.date()),
            "policy": policy, "allocations": weights, "previous_allocations": previous,
            "traceability": response.get("traceability", {}), "tool_calls": tool_trace,
            "gross_return": gross, "turnover": turnover, "metrics_snapshot": metrics, "news_snapshot": headlines,
            "asset_returns": {k: float(v) for k, v in asset_returns.items()},
            "next_policy": policy,
        }
        records.append(record)
        evolves = method in {"EvolveBase", "EvolveStrategy", "EvolveTrade"}
        new_in_batch = [r for r in records if r["method"] == method][-update_interval:]
        if evolves and len(records) % update_interval == 0:
            policy, update_trace = update_policy(client, method, policy, new_in_batch)
            record["policy_update"] = update_trace
            record["next_policy"] = policy
        previous = weights
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    frame = pd.DataFrame(records)
    gross = pd.Series(frame["gross_return"].values, index=pd.to_datetime(frame["feedback_date"]))
    net = gross - frame.set_index(pd.to_datetime(frame["feedback_date"]))["turnover"] * config["transaction_cost_bps"] / 10000
    row = {"method": method, "period": period, "seed": seed, "update_interval": update_interval, "protocol": run_tag or "legacy", **evaluate(gross)}
    row.update({f"net_{k}": v for k, v in evaluate(net).items()})
    row["turnover"] = float(frame["turnover"].sum())
    row["average_cash"] = float(frame["allocations"].map(lambda x: x["CASH"]).mean())
    pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
    return row

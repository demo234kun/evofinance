import json
import http.client
import os
import re
import ssl
import time
import urllib.error
import urllib.request
import numpy as np
import pandas as pd


def _json_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("DeepSeek response has no JSON object")
    return json.loads(text[start : end + 1])


class DeepSeekClient:
    def __init__(self, model: str = "deepseek-chat"):
        self.api_key = os.environ["DEEPSEEK_API_KEY"]
        self.model = model

    def complete(self, system: str, user: str, retries: int = 8) -> dict:
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system + "\nRespond with one valid JSON object."}, {"role": "user", "content": user}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return _json_object(body["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt + 1 == retries or exc.code < 500:
                    raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
                time.sleep(min(60, 2 ** attempt))
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError, http.client.IncompleteRead, http.client.RemoteDisconnected, ssl.SSLError, ConnectionError):
                if attempt + 1 == retries:
                    raise
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError("unreachable")

    def tool_turn(self, messages: list[dict], tools: list[dict], retries: int = 8, tool_choice: str = "auto") -> dict:
        payload = json.dumps({"model": self.model, "temperature": 0.1, "messages": messages, "tools": tools, "tool_choice": tool_choice}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt + 1 == retries or exc.code < 500:
                    raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
                time.sleep(min(60, 2 ** attempt))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, http.client.IncompleteRead, http.client.RemoteDisconnected, ssl.SSLError, ConnectionError):
                if attempt + 1 == retries:
                    raise
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError("unreachable")


def tool_snapshot(history: pd.DataFrame) -> dict:
    returns = history.pct_change()
    last = history.iloc[-1]
    return {
        "get_price": {
            "as_of": str(history.index[-1].date()),
            "close": {k: round(float(v), 4) for k, v in last.items()},
        },
        "code_interpreter": {
            "return_5d": {k: round(float(v), 6) for k, v in (last / history.iloc[-6] - 1).items()},
            "return_20d": {k: round(float(v), 6) for k, v in (last / history.iloc[-21] - 1).items()},
            "volatility_20d": {k: round(float(v), 6) for k, v in (returns.tail(20).std() * np.sqrt(252)).items()},
            "drawdown_60d": {k: round(float(v), 6) for k, v in (last / history.tail(60).max() - 1).items()},
        },
        "news_searcher": {"cutoff": "t-1", "items": [], "status": "no point-in-time news in base smoke dataset"},
    }


class DeepSeekTradingAgent:
    def __init__(self, client: DeepSeekClient, max_weight: float):
        self.client = client
        self.max_weight = max_weight

    def decide(self, history: pd.DataFrame, policy: str) -> tuple[pd.Series, dict]:
        tools = tool_snapshot(history)
        system = policy + " Return strict JSON only. Use only supplied point-in-time tool results."
        user = json.dumps(
            {
                "task": "Allocate long-only weights across assets and CASH for the next trading day.",
                "constraints": {"sum": 1.0, "max_asset_weight": self.max_weight, "cash_allowed": True},
                "tool_results": tools,
                "output_schema": {"weights": {"TICKER": 0.0, "CASH": 0.0}, "reasoning": {}, "checks": []},
            },
            ensure_ascii=False,
        )
        result = self.client.complete(system, user)
        assets = list(history.columns)
        raw = {a: max(0.0, min(self.max_weight, float(result.get("weights", {}).get(a, 0.0)))) for a in assets}
        raw["CASH"] = max(0.0, float(result.get("weights", {}).get("CASH", 0.0)))
        total = sum(raw.values())
        if total <= 0:
            raw = {a: 0.0 for a in assets} | {"CASH": 1.0}
        else:
            raw = {k: v / total for k, v in raw.items()}
        return pd.Series(raw), {"tool_results": tools, "model_output": result}


class DeepSeekPolicyAgent:
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def refine(self, policy: str, records: list[dict]) -> str:
        result = self.client.complete(
            "You are the EvolveTrade Policy Agent. Revise the reusable trading procedure, not individual trades. Preserve useful rules, correct evidenced failures, and avoid hindsight leakage. Return JSON only.",
            json.dumps(
                {
                    "current_policy": policy,
                    "completed_experience_batch": records,
                    "output_schema": {"diagnosis": [], "preserved_rules": [], "updated_policy": "string"},
                },
                ensure_ascii=False,
            ),
        )
        updated = str(result.get("updated_policy", "")).strip()
        return updated or policy


import csv
import datetime as dt
import json
from pathlib import Path
import time
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import pandas as pd


def _epoch(value: str) -> int:
    return int(dt.datetime.fromisoformat(value).replace(tzinfo=dt.timezone.utc).timestamp())


def yahoo_prices(ticker: str, start: str, end: str) -> pd.Series:
    query = urllib.parse.urlencode({"period1": _epoch(start), "period2": _epoch(end), "interval": "1d", "events": "history", "includeAdjustedClose": "true"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    values = adjclose or result["indicators"]["quote"][0]["close"]
    return pd.Series(values, index=dates, name=ticker, dtype=float).dropna()


def download_price_panel(config: dict, output: Path) -> pd.DataFrame:
    tickers = set(config["assets"] + [config["benchmark"]])
    cache = output.parent / "hf_monthly"
    cache.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(config["price_start"], config["price_end"], freq="M")
    frames = []
    for month in months:
        path = cache / f"train-{month}.parquet"
        if not path.exists():
            url = f"https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price/resolve/main/data/train-{month}.parquet?download=true"
            error = None
            for attempt in range(6):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        payload = response.read()
                    if len(payload) < 10000:
                        raise ValueError("downloaded parquet is too small")
                    path.write_bytes(payload)
                    error = None
                    break
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    error = exc
                    time.sleep(3 * (attempt + 1))
            if error is not None:
                raise error
        frame = pd.read_parquet(path)
        frames.append(frame[frame["symbol"].isin(tickers)])
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= config["price_start"]) & (raw["date"] < config["price_end"])]
    panel = raw.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last").sort_index().ffill().dropna()
    missing = tickers.difference(panel.columns)
    if missing:
        raise ValueError(f"missing symbols: {sorted(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output)
    raw.sort_values(["date", "symbol"]).to_parquet(output.with_name("us_daily_ohlcv.parquet"), index=False)
    return panel


def gdelt_news(ticker: str, company: str, start: str, end: str, max_records: int = 50) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "query": f'("{company}" OR {ticker}) sourcecountry:US',
            "mode": "ArtList",
            "maxrecords": max_records,
            "format": "json",
            "startdatetime": start.replace("-", "") + "000000",
            "enddatetime": end.replace("-", "") + "235959",
            "sort": "HybridRel",
        }
    )
    request = urllib.request.Request(f"https://api.gdeltproject.org/api/v2/doc/doc?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [{"ticker": ticker, **item} for item in body.get("articles", [])]


def save_news_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ticker", "seendate", "title", "url", "domain", "language", "sourcecountry"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def download_newsmood(output: Path) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        url = "https://huggingface.co/datasets/thorwhalen/newsmood-data/resolve/main/news.parquet?download=true"
        error = None
        for attempt in range(8):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = response.read()
                if len(payload) < 100_000_000:
                    raise ValueError("news parquet is incomplete")
                output.write_bytes(payload)
                error = None
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                error = exc
                time.sleep(5 * (attempt + 1))
        if error is not None:
            raise error
    frame = pd.read_parquet(output)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert(None)
    return frame


COMPANIES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "JPM": "JPMorgan Chase", "V": "Visa",
    "JNJ": "Johnson Johnson", "UNH": "UnitedHealth", "PG": "Procter Gamble", "KO": "Coca Cola", "XOM": "Exxon Mobil",
    "CAT": "Caterpillar", "WMT": "Walmart", "META": "Meta Platforms", "TSLA": "Tesla", "AMZN": "Amazon"
}


def google_news_rss(ticker: str, company: str, start: str, end: str) -> list[dict]:
    query = urllib.parse.quote(f'("{company}" OR {ticker}) stock after:{start} before:{end}')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                root = ET.fromstring(response.read())
            rows = []
            for item in root.findall(".//item"):
                source = item.find("source")
                rows.append(
                    {
                        "doc_id": str(abs(hash((ticker, item.findtext("link"))))),
                        "ts": parsedate_to_datetime(item.findtext("pubDate")).isoformat(),
                        "source": source.text if source is not None else "Google News",
                        "query": ticker,
                        "title": item.findtext("title"),
                        "description": item.findtext("description"),
                        "link": item.findtext("link"),
                    }
                )
            return rows
        except Exception as exc:
            error = exc
            time.sleep(3 * (attempt + 1))
    raise error


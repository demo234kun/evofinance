import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.real_data import COMPANIES, google_news_rss

parser = argparse.ArgumentParser()
parser.add_argument("--periods", nargs="+", default=["2025-01"])
args = parser.parse_args()
config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
cache = ROOT / "data" / "real" / "news" / "google_cache"
cache.mkdir(parents=True, exist_ok=True)
for period in args.periods:
    start, end = config["evaluation_windows"][period]
    for ticker, company in COMPANIES.items():
        path = cache / f"{period}_{ticker}.json"
        if path.exists():
            continue
        rows = google_news_rss(ticker, company, start, end)
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(period, ticker, len(rows), flush=True)
rows = []
for path in cache.glob("*.json"):
    period, ticker = path.stem.split("_", 1)
    rows.extend({"period": period, **row} for row in json.loads(path.read_text(encoding="utf-8")))
frame = pd.DataFrame(rows)
frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
frame = frame.drop_duplicates(["query", "ts", "title"]).sort_values(["ts", "query"])
output = ROOT / "data" / "real" / "news" / "google_news_historical.parquet"
frame.to_parquet(output, index=False)
print(frame.shape, frame["ts"].min(), frame["ts"].max())

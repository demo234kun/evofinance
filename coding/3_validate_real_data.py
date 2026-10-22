import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
prices = pd.read_csv(ROOT / "data" / "real" / "us_daily_adjusted_close.csv", index_col=0, parse_dates=True)
news = pd.read_parquet(ROOT / "data" / "real" / "news" / "newsmood_news.parquet")
google_path = ROOT / "data" / "real" / "news" / "google_news_historical.parquet"
google = pd.read_parquet(google_path) if google_path.exists() else pd.DataFrame()
checks = {
    "price_rows": len(prices), "price_assets": prices.shape[1], "price_start": str(prices.index.min().date()), "price_end": str(prices.index.max().date()),
    "price_duplicate_dates": int(prices.index.duplicated().sum()), "price_missing_values": int(prices.isna().sum().sum()),
    "price_nonpositive_values": int((prices <= 0).sum().sum()), "price_nonfinite_values": int((~np.isfinite(prices)).sum().sum()),
    "news_rows": len(news), "news_start": str(pd.to_datetime(news["ts"], utc=True).min()), "news_end": str(pd.to_datetime(news["ts"], utc=True).max()),
    "news_duplicate_doc_ids": int(news["doc_id"].duplicated().sum()), "news_missing_titles": int(news["title"].isna().sum()),
    "google_news_rows": len(google),
    "google_news_start": None if google.empty else str(pd.to_datetime(google["ts"], utc=True).min()),
    "google_news_end": None if google.empty else str(pd.to_datetime(google["ts"], utc=True).max()),
}
for period, (start, end) in config["evaluation_windows"].items():
    checks[f"trading_days_{period}"] = len(prices.loc[start:end])
out = ROOT / "result" / "data_quality.json"
out.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(checks, ensure_ascii=False, indent=2))


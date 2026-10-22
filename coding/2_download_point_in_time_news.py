import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.real_data import download_newsmood

frame = download_newsmood(ROOT / "data" / "real" / "news" / "newsmood_news.parquet")
print(frame.shape, frame["ts"].min(), frame["ts"].max())

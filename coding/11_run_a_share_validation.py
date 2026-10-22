import json
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evolvetrade.paper_runner as runner

cfg = json.loads((ROOT / "config" / "a_share_config.json").read_text(encoding="utf-8"))
prices = pd.read_csv(ROOT / "data" / "a_share" / "a_share_daily_adjusted_close.csv", index_col=0, parse_dates=True)
news = pd.read_parquet(ROOT / "data" / "a_share" / "point_in_time_news.parquet")
news["ts"] = pd.to_datetime(news["ts"])
assets = pd.read_csv(ROOT / "data" / "a_share" / "selected_assets.csv")["symbol"].tolist()
cfg["assets"] = assets
runner.COMPANIES = {symbol: symbol for symbol in assets}
runner.load_real_data = lambda root: (prices, news)

rows = []
for period in cfg["evaluation_windows"]:
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            for attempt in range(6):
                try:
                    row = runner.run_llm_method(ROOT, cfg, method, period, seed, 5, "a_share_protocol_v2")
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(min(300, 30 * (2 ** attempt)))
            rows.append(row)
            pd.DataFrame(rows).to_csv(ROOT / "result" / "a_share_llm_results.csv", index=False)

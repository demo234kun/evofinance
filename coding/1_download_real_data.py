import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.real_data import download_price_panel

config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
panel = download_price_panel(config, ROOT / "data" / "real" / "us_daily_adjusted_close.csv")
print(panel.shape, panel.index.min().date(), panel.index.max().date())


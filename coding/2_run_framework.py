import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.data import load_prices
from evolvetrade.experiment import run_experiment

config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
prices = load_prices(ROOT / "data" / "demo" / "prices.csv")
print(run_experiment(prices, config, ROOT / "result").to_string(index=False))


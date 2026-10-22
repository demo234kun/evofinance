import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.data import generate_demo

config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
generate_demo(config, ROOT / "data" / "demo" / "prices.csv")


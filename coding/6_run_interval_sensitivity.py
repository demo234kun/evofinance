import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.paper_runner import run_llm_method

config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
rows = []
for interval in [1, 3, 5, 7]:
    for period in ["2025-01", "2025-04", "2025-09"]:
        for seed in config["seeds"]:
            rows.append(run_llm_method(ROOT, config, "EvolveTrade", period, seed, interval, f"protocol_v2_N{interval}"))
pd.DataFrame(rows).to_csv(ROOT / "result" / "interval_sensitivity.csv", index=False)


import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.paper_runner import run_llm_method

config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
config["evaluation_windows"]["long_50d"] = config["long_horizon"]
rows = []
for method in config["methods"]:
    for seed in config["seeds"]:
        rows.append(run_llm_method(ROOT, config, method, "long_50d", seed, 5, "protocol_v2_paper_50d"))
pd.DataFrame(rows).to_csv(ROOT / "result" / "long_horizon.csv", index=False)


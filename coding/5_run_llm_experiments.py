import argparse
import json
import sys
import time
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolvetrade.paper_runner import run_llm_method

parser = argparse.ArgumentParser()
parser.add_argument("--period", default="all")
parser.add_argument("--method", default="all")
parser.add_argument("--seed", type=int)
args = parser.parse_args()
config = json.loads((ROOT / "config" / "base_config.json").read_text(encoding="utf-8"))
periods = list(config["evaluation_windows"]) if args.period == "all" else [args.period]
methods = config["methods"] if args.method == "all" else [args.method]
seeds = [args.seed] if args.seed is not None else config["seeds"]
rows = []
total = len(periods) * len(methods) * len(seeds)
completed = 0
print(f"主矩阵: 0/{total} (0.0%)")
for period in periods:
    for method in methods:
        for seed in seeds:
            for attempt in range(6):
                try:
                    row = run_llm_method(ROOT, config, method, period, seed, 5, "protocol_v2")
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(min(300, 30 * (2 ** attempt)))
            rows.append(row)
            completed += 1
            print(f"主矩阵: {completed}/{total} ({completed / total:.1%}) 当前={period}/{method}/{seed}", flush=True)
            print(json.dumps(row, ensure_ascii=False))
output = ROOT / "result" / "llm_main_results.csv"
current = pd.DataFrame(rows)
if output.exists():
    current = pd.concat([pd.read_csv(output), current], ignore_index=True)
if "protocol" not in current.columns:
    current["protocol"] = "legacy"
current["protocol"] = current["protocol"].fillna("legacy")
current = current.drop_duplicates(["method", "period", "seed", "update_interval", "protocol"], keep="last")
current.to_csv(output, index=False)

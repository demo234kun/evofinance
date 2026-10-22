import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("data_audit", "3_validate_real_data.py"),
    ("innovation_smoke", "15_innovation_full_smoke.py"),
    ("formal_matrix", "5_run_llm_experiments.py"),
    ("interval_sensitivity", "6_run_interval_sensitivity.py"),
    ("long_horizon", "7_run_long_horizon.py"),
    ("paper_tables", "8_generate_paper_tables.py"),
    ("paper_figures", "9_generate_paper_figures.py"),
    ("a_share_validation", "11_run_a_share_validation.py"),
    ("a_share_results", "13_generate_a_share_results.py"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--from-stage", default="data_audit", choices=[x[0] for x in STAGES])
args = parser.parse_args()
start = [x[0] for x in STAGES].index(args.from_stage)
env = os.environ.copy()
env["PYTHONUTF8"] = "1"
for index, (name, script) in enumerate(STAGES[start:], start + 1):
    print(f"后处理: {index}/{len(STAGES)} ({index / len(STAGES):.1%}) 当前={name}", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / "coding" / script)], cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"阶段失败: {name}，返回码={result.returncode}")
print("后处理: 100.0% 全部完成", flush=True)

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"]))


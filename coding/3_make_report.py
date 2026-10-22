import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
returns = pd.read_csv(ROOT / "result" / "evolvetrade_daily_returns.csv", index_col=0, parse_dates=True)["return"]
equity = (1 + returns).cumprod()
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(equity.index, equity.values, color="#174A7E", linewidth=1.8)
ax.set_title("EvolveTrade Clean-room Smoke Test")
ax.set_ylabel("Net Value")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "result" / "evolvetrade_equity.png", dpi=180)
plt.close(fig)


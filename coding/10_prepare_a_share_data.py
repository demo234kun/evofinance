import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "a_share_config.json").read_text(encoding="utf-8"))
SOURCE = Path(CFG["source_root"])
OUT = ROOT / "data" / "a_share"
OUT.mkdir(parents=True, exist_ok=True)

membership = pd.read_parquet(CFG["membership_file"])
csi300 = set(membership.loc[membership["index_name"] == "CSI300", "symbol"])
frames = []
for path in sorted(SOURCE.glob("dt=*/data.parquet")):
    date = pd.Timestamp(path.parent.name.split("=")[1])
    if date < pd.Timestamp(CFG["selection_start"]):
        continue
    frame = pd.read_parquet(path, columns=["symbol", "time", "close", "amount"])
    frames.append(frame[frame["symbol"].isin(csi300)])
panel = pd.concat(frames, ignore_index=True)
panel["time"] = pd.to_datetime(panel["time"])
selection = panel[(panel["time"] >= CFG["selection_start"]) & (panel["time"] <= CFG["selection_end"])]
coverage = selection.groupby("symbol")["time"].nunique()
liquidity = selection.groupby("symbol")["amount"].median()
eligible = coverage[coverage >= coverage.max() * 0.95].index
assets = liquidity.reindex(eligible).nlargest(CFG["asset_count"]).index.tolist()
prices = panel[panel["symbol"].isin(assets)].pivot(index="time", columns="symbol", values="close").sort_index()
prices = prices.loc[:, prices.notna().mean().sort_values(ascending=False).index].ffill()
prices["CSI300_EW"] = panel.pivot(index="time", columns="symbol", values="close").sort_index().pct_change(fill_method=None).mean(axis=1).fillna(0).add(1).cumprod()
prices.to_csv(OUT / "a_share_daily_adjusted_close.csv")
pd.DataFrame({"symbol": assets, "selection_metric": "median_amount_2024"}).to_csv(OUT / "selected_assets.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(columns=["ts", "title", "description", "source"]).to_parquet(OUT / "point_in_time_news.parquet", index=False)
print(json.dumps({"assets": assets, "dates": len(prices), "start": str(prices.index.min().date()), "end": str(prices.index.max().date())}, ensure_ascii=False))

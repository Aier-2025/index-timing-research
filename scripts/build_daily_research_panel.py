"""Build a date-union research panel with conservative factor admission rules."""

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(r"F:\data\index_timing_raw")
FACTOR_DIR = Path(r"F:\量化因子库\docs\factor_library")
INDEX_PANEL = DATA_DIR / "index_timing_daily_panel.csv"
OUT = DATA_DIR / "index_timing_research_panel.csv"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "index_universe.json"


def main():
    prices = pd.read_csv(INDEX_PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    prices = prices.sort_values(["index_code", "trade_date"])
    # The raw union panel deliberately contains pre-inception placeholder rows
    # for indices that did not yet exist. They remain in the raw audit, but
    # must never enter feature, return, label, or equity-curve calculations.
    prices = prices.loc[prices["open"].notna() & prices["close"].notna()].copy()
    grouped = prices.groupby("index_code", group_keys=False)
    prices["ret_1d_close_to_close"] = grouped["close"].pct_change(fill_method=None)
    prices["next_open_to_open"] = grouped["open"].shift(-2).div(grouped["open"].shift(-1)).sub(1)
    prices["execution_date"] = grouped["trade_date"].shift(-1)
    prices["signal_date"] = prices["trade_date"]

    # All technical features use observations through the signal date only.
    for horizon in [1, 3, 5, 10, 20, 60, 120, 240]:
        prices[f"mom_{horizon}d"] = grouped["close"].pct_change(horizon, fill_method=None)
        prices[f"close_to_ma_{horizon}d"] = prices["close"].div(
            grouped["close"].transform(lambda s: s.rolling(horizon, min_periods=horizon).mean())
        ).sub(1)
        prices[f"vol_{horizon}d"] = grouped["ret_1d_close_to_close"].transform(
            lambda s: s.rolling(horizon, min_periods=horizon).std() * (252 ** 0.5)
        )
        prices[f"drawdown_{horizon}d"] = prices["close"].div(
            grouped["close"].transform(lambda s: s.rolling(horizon, min_periods=horizon).max())
        ).sub(1)
    prices["intraday_range"] = prices["high"].sub(prices["low"]).div(prices["close"])
    prices["close_location"] = prices["close"].sub(prices["low"]).div(prices["high"].sub(prices["low"]))
    prices["macro_sentiment_admitted"] = prices["trade_date"].ge(pd.Timestamp("2015-01-01"))

    factor_panel = FACTOR_DIR / "timing_feature_panel_2005_2026.csv"
    factor_status = {"loaded": False, "path": str(factor_panel), "reason": "not loaded"}
    if factor_panel.exists():
        factors = pd.read_csv(factor_panel, parse_dates=["date"])
        # Only retain fields whose names are not index price/return targets. The
        # separate factor contract still controls source lag and admission.
        factors = factors.rename(columns={"date": "trade_date"})
        drop = {"csi1000", "hs300", "sz50", "zz500"}
        drop.update(column for column in factors.columns if column.endswith("_ret1"))
        factors = factors.drop(columns=[c for c in drop if c in factors.columns])
        factors = factors.loc[factors["trade_date"].ge("2015-01-01")]
        prices = prices.merge(factors, on="trade_date", how="left", suffixes=("", "_factor"))
        factor_status = {"loaded": True, "path": str(factor_panel), "columns": int(factors.shape[1] - 1), "first_admitted_date": "2015-01-01"}
    prices.to_csv(OUT, index=False, encoding="utf-8-sig")
    manifest = {"output": str(OUT), "rows": int(len(prices)), "date_min": prices.trade_date.min().date().isoformat(), "date_max": prices.trade_date.max().date().isoformat(), "factor_status": factor_status, "config": json.loads(CONFIG.read_text(encoding="utf-8"))}
    (OUT.with_suffix(".manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Preliminary long-short analysis from frozen rolling OOS predictions."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

DATA = Path(r"F:\data\index_timing_raw")
CONFIGS = {
    "000905": DATA / "index_oos_workers_v2" / "000905" / "oos_signal.csv",
    "000852": DATA / "index_oos_workers_v2" / "000852" / "oos_signal.csv",
    "399006": DATA / "index_oos_workers_v2" / "399006" / "oos_signal.csv",
    "000688": DATA / "index_oos_workers_v2" / "000688" / "oos_signal.csv",
    "932000": DATA / "index_oos_windows" / "932000_2025" / "932000" / "oos_signal.csv",
}
NAMES = {"000905": "中证500", "000852": "中证1000", "399006": "创业板指", "000688": "科创50", "932000": "中证2000"}
COST = 0.0008


def metrics(data):
    ret = data["long_short_return"]
    active = data["long_short_return"] - data["benchmark_return"]
    beta = data["long_short_return"].rolling(252, min_periods=60).cov(data["benchmark_return"]) / data["benchmark_return"].rolling(252, min_periods=60).var()
    beta = beta.shift(1).fillna(0.0)
    beta_neutral = data["long_short_return"] - beta * data["benchmark_return"]
    equity = (1 + ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "rows": int(len(data)),
        "start": data.trade_date.min().date().isoformat(),
        "end": data.trade_date.max().date().isoformat(),
        "annualized_return": float((1 + ret).prod() ** (252 / len(ret)) - 1),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(252)) if ret.std(ddof=1) else None,
        "annualized_excess": float((1 + active).prod() ** (252 / len(active)) - 1),
        "active_sharpe": float(active.mean() / active.std(ddof=1) * np.sqrt(252)) if active.std(ddof=1) else None,
        "beta_neutral_annualized_excess": float((1 + beta_neutral).prod() ** (252 / len(beta_neutral)) - 1),
        "beta_neutral_sharpe": float(beta_neutral.mean() / beta_neutral.std(ddof=1) * np.sqrt(252)) if beta_neutral.std(ddof=1) else None,
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(data.turnover.sum() / len(data) * 252),
        "long_ratio": float((data.position > 0).mean()),
        "short_ratio": float((data.position < 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "reports" / "long_short_oos_analysis")
    args = parser.parse_args()
    rows = []
    ledgers = []
    for code, path in CONFIGS.items():
        raw = pd.read_csv(path, parse_dates=["trade_date"])
        for horizon, data in raw.groupby("horizon_days"):
            data = data.dropna(subset=["prediction", "benchmark_return"]).copy().sort_values("trade_date")
            if data.empty:
                continue
            data["position"] = np.where(data["prediction"] >= 0, 1.0, -1.0)
            data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())
            data["long_short_return"] = data["position"] * data["benchmark_return"] - data["turnover"] * COST
            result = metrics(data)
            result.update({"index_code": code, "index_name": NAMES[code], "horizon_days": int(horizon)})
            rows.append(result)
            ledgers.append(data[["trade_date", "index_code", "horizon_days", "prediction", "position", "benchmark_return", "turnover", "long_short_return"]])
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(ledgers, ignore_index=True).to_csv(out / "ledger.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

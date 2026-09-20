"""Run a fixed, non-optimized price-only timing baseline for all five indices."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
OUT_LEDGER = DATA_DIR / "price_technical_baseline_ledger.csv"
OUT_SUMMARY = DATA_DIR / "price_technical_baseline_summary.json"
COMMISSION_ONE_WAY = 0.0003
SLIPPAGE_ONE_WAY = 0.0005
ONE_WAY_COST = COMMISSION_ONE_WAY + SLIPPAGE_ONE_WAY


def position(score):
    return np.select([score >= 0.67, score <= -0.67], [1.0, 0.0], default=0.5)


def metrics(data):
    data = data.dropna(subset=["strategy_return", "benchmark_return"])
    if data.empty:
        return {"rows": 0}
    r = data["strategy_return"]
    bench = data["benchmark_return"]
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    active = r - bench
    return {
        "rows": int(len(data)),
        "annualized_return": float((1 + r).prod() ** (252 / len(r)) - 1),
        "benchmark_annualized_return": float((1 + bench).prod() ** (252 / len(bench)) - 1),
        "annualized_excess": float(((1 + (r - bench)).cumprod().iloc[-1]) ** (252 / len(r)) - 1.0),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) else None,
        "active_sharpe": float(active.mean() / active.std(ddof=1) * np.sqrt(252)) if active.std(ddof=1) else None,
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(data["target_position"].diff().abs().fillna(0).sum() / len(data) * 252),
        "total_commission": float(data["commission"].sum()),
        "total_slippage": float(data["slippage"].sum()),
        "total_cost": float(data["total_cost"].sum()),
    }


def main():
    frame = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    frame = frame.sort_values(["index_code", "trade_date"])
    rows = []
    for code, group in frame.groupby("index_code", sort=False):
        group = group.copy()
        features = group[["mom_20d", "close_to_ma_60d", "mom_120d"]]
        eligible = features.notna().all(axis=1)
        score = features.gt(0).sum(axis=1)
        score = (score - 1.5) / 1.5
        group["score"] = score
        group["target_position"] = pd.Series(position(score), index=group.index).where(eligible)
        group["turnover"] = group["target_position"].diff().abs().fillna(0.0)
        group["benchmark_return"] = group["next_open_to_open"]
        group["commission"] = group["turnover"] * COMMISSION_ONE_WAY
        group["slippage"] = group["turnover"] * SLIPPAGE_ONE_WAY
        group["total_cost"] = group["commission"] + group["slippage"]
        group["gross_strategy_return"] = group["target_position"] * group["benchmark_return"]
        group["strategy_return"] = group["gross_strategy_return"] - group["total_cost"]
        group["index_name"] = code
        rows.append(group[["trade_date", "index_code", "index_name", "score", "target_position", "turnover", "commission", "slippage", "total_cost", "gross_strategy_return", "benchmark_return", "strategy_return"]])
    ledger = pd.concat(rows, ignore_index=True).dropna(subset=["benchmark_return"])
    summary = {code: metrics(group) for code, group in ledger.groupby("index_code")}
    summary["protocol"] = {"signal": "T close", "execution": "T+1 open", "holding_return": "T+1 open to T+2 open", "commission_one_way": COMMISSION_ONE_WAY, "slippage_one_way": SLIPPAGE_ONE_WAY, "total_cost_one_way": ONE_WAY_COST, "rule": "equal vote of fixed 20d momentum, 60d close/MA, 120d momentum"}
    ledger.to_csv(OUT_LEDGER, index=False, encoding="utf-8-sig")
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Search a small, pre-defined strategy library per index.

The search is limited to train/development and validation intervals. It never
uses the future frozen test interval, and it reports only concatenated interval
performance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
PROFILES = Path(__file__).resolve().parents[1] / "config" / "index_research_profiles.json"
OUT_ROOT = DATA_DIR / "index_strategy_search"
COMMISSION_ONE_WAY = 0.0003
SLIPPAGE_ONE_WAY = 0.0005
ONE_WAY_COST = COMMISSION_ONE_WAY + SLIPPAGE_ONE_WAY


def metrics(frame):
    frame = frame.dropna(subset=["strategy_return", "benchmark_return"])
    if frame.empty:
        return {"rows": 0, "annualized_excess": np.nan, "active_sharpe": np.nan}
    ret = frame["strategy_return"]
    bench = frame["benchmark_return"]
    active = ret - bench
    return {
        "rows": int(len(frame)),
        "annualized_return": float((1 + ret).prod() ** (252 / len(ret)) - 1),
        "benchmark_annualized_return": float((1 + bench).prod() ** (252 / len(bench)) - 1),
        "annualized_excess": float(((1 + (ret - bench)).cumprod().iloc[-1]) ** (252 / len(ret)) - 1.0),
        "active_sharpe": float(active.mean() / active.std(ddof=1) * np.sqrt(252)) if active.std(ddof=1) else np.nan,
        "max_drawdown": float(((1 + ret).cumprod() / (1 + ret).cumprod().cummax() - 1).min()),
        "annual_turnover": float(frame["turnover"].sum() / len(frame) * 252),
        "total_commission": float(frame["commission"].sum()),
        "total_slippage": float(frame["slippage"].sum()),
        "total_cost": float(frame["total_cost"].sum()),
    }


def strategy_score(group, name):
    f = group
    if name == "mom_fast":
        return f["mom_20d"]
    if name == "mom_mid":
        return f["mom_60d"]
    if name == "mom_slow":
        return f["mom_120d"]
    if name == "trend_vote":
        return f[["mom_20d", "mom_60d", "close_to_ma_60d", "drawdown_60d"]].mean(axis=1)
    if name == "slow_trend_vote":
        return f[["mom_60d", "mom_120d", "mom_240d", "close_to_ma_120d"]].mean(axis=1)
    if name == "reversal_vote":
        return -f[["mom_20d", "mom_60d", "close_to_ma_60d", "drawdown_60d"]].mean(axis=1)
    if name == "volatility_defensive":
        score = f["mom_60d"].gt(0).astype(float)
        score = score.where(f["vol_20d"].le(f["vol_20d"].rolling(252, min_periods=60).median()), 0.5)
        return score - 0.5
    if name == "drawdown_defensive":
        return pd.Series(np.where((f["mom_120d"] > 0) & (f["drawdown_120d"] > -0.20), 0.5, -0.5), index=f.index)
    raise ValueError(name)


def position_from_score(score, direction=1.0, threshold=0.0):
    score = pd.Series(score) * direction
    return pd.Series(np.select([score > threshold, score < -threshold], [1.0, 0.0], default=0.5), index=score.index)


def evaluate(group, score, holding, start, end):
    data = group.loc[(group.index >= pd.Timestamp(start)) & (group.index <= pd.Timestamp(end))].copy()
    score = score.reindex(data.index)
    eligible = score.notna()
    target = position_from_score(score, threshold=0.0).where(eligible)
    target = target.where(np.arange(len(target)) % holding == 0).ffill().fillna(0.5)
    data["target_position"] = target
    data["benchmark_return"] = data["next_open_to_open"]
    data["turnover"] = data["target_position"].diff().abs().fillna(data["target_position"].abs())
    data["commission"] = data["turnover"] * COMMISSION_ONE_WAY
    data["slippage"] = data["turnover"] * SLIPPAGE_ONE_WAY
    data["total_cost"] = data["commission"] + data["slippage"]
    data["gross_strategy_return"] = data["target_position"] * data["benchmark_return"]
    data["strategy_return"] = data["gross_strategy_return"] - data["total_cost"]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, choices=["000905", "000852", "399006", "000688", "932000"])
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    profile = profiles[args.index]
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    group = panel.loc[panel["index_code"].eq(args.index)].sort_values("trade_date").set_index("trade_date")
    first_valid = group.index[group["close"].notna()].min()
    development_start = group.index[group.index.get_loc(first_valid) + int(profile["oos"]["train_days"]) + max(profile["oos"]["horizons_days"])]
    development_end = pd.Timestamp(profile["oos"]["start"]) - pd.Timedelta(days=1)
    backtest_start = pd.Timestamp(profile["oos"]["start"])
    backtest_end = group.index.max()
    strategies = ["mom_fast", "mom_mid", "mom_slow", "trend_vote", "slow_trend_vote", "reversal_vote", "volatility_defensive", "drawdown_defensive"]
    rows = []
    ledgers = []
    for name in strategies:
        score = strategy_score(group, name)
        for holding in profile["oos"]["horizons_days"]:
            train = evaluate(group, score, holding, development_start, development_end)
            valid = evaluate(group, score, holding, backtest_start, backtest_end)
            train_metrics = metrics(train)
            valid_metrics = metrics(valid)
            row = {"index_code": args.index, "index_name": profile["name"], "strategy": name, "holding_days": holding,
                   "training_start": str(development_start.date()), "training_end": str(development_end.date()),
                   "backtest_start": str(backtest_start.date()), "backtest_end": str(backtest_end.date()),
                   "training_annualized_excess": train_metrics["annualized_excess"], "training_active_sharpe": train_metrics["active_sharpe"],
                   "backtest_annualized_excess": valid_metrics["annualized_excess"], "backtest_active_sharpe": valid_metrics["active_sharpe"],
                   "training_rows": train_metrics["rows"], "backtest_rows": valid_metrics["rows"],
                   "training_annual_turnover": train_metrics["annual_turnover"], "backtest_annual_turnover": valid_metrics["annual_turnover"]}
            row["passes_dual_interval_gate"] = bool(
                row["training_annualized_excess"] > 0 and row["training_active_sharpe"] > 0 and
                row["backtest_annualized_excess"] > 0 and row["backtest_active_sharpe"] > 0 and
                row["training_annualized_excess"] >= row["backtest_annualized_excess"] and
                row["training_active_sharpe"] >= row["backtest_active_sharpe"]
            )
            row["gate_score"] = float(min(row["training_annualized_excess"], row["backtest_annualized_excess"], row["training_active_sharpe"], row["backtest_active_sharpe"]))
            rows.append(row)
            train.assign(period="training").reset_index().to_csv(OUT_ROOT / "_tmp_train.csv", index=False) if False else None
            ledgers.append(train.assign(period="training", strategy=name, holding_days=holding).reset_index())
            ledgers.append(valid.assign(period="backtest", strategy=name, holding_days=holding).reset_index())
    out = OUT_ROOT / args.index
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["passes_dual_interval_gate", "gate_score"], ascending=[False, False])
    result.to_csv(out / "strategy_search.csv", index=False, encoding="utf-8-sig")
    pd.concat(ledgers, ignore_index=True).to_csv(out / "strategy_ledgers.csv", index=False, encoding="utf-8-sig")
    summary = {"index": args.index, "index_name": profile["name"], "training_interval": [str(development_start.date()), str(development_end.date())], "backtest_interval": [str(validation_start.date()), str(validation_end.date())], "passing_strategies": int(result["passes_dual_interval_gate"].sum()), "best": result.iloc[0].to_dict() if len(result) else None, "backtest_status": "not_run"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

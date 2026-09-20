"""Search a small index-specific rule library with explicit trading costs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import screen_index_timing_factors as screen


DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
CONTRACT = Path(r"F:\量化因子库\docs\reports\csi1000_factor_term_audit_20260819\feature_contract_draft.csv")
STATE_QUALITY = Path(r"F:\量化因子库\docs\factor_library\market_state_factor_quality.csv")
PROFILES = Path(__file__).resolve().parents[1] / "config" / "index_research_profiles.json"
OUT = DATA_DIR / "index_targeted_strategy_search"
COMMISSION = 0.0003
SLIPPAGE = 0.0005
TOTAL_COST = COMMISSION + SLIPPAGE


def performance(data):
    data = data.dropna(subset=["strategy_return", "benchmark_return"])
    if data.empty:
        return {"rows": 0, "annualized_excess": np.nan, "active_sharpe": np.nan}
    ret, bench = data.strategy_return, data.benchmark_return
    active = ret - bench
    return {
        "rows": int(len(data)),
        "annualized_return": float((1 + ret).prod() ** (252 / len(ret)) - 1),
        "benchmark_annualized_return": float((1 + bench).prod() ** (252 / len(bench)) - 1),
        "annualized_excess": float(((1 + (ret - bench)).cumprod().iloc[-1]) ** (252 / len(ret)) - 1.0),
        "active_sharpe": float(active.mean() / active.std(ddof=1) * np.sqrt(252)) if active.std(ddof=1) else np.nan,
        "max_drawdown": float(((1 + ret).cumprod() / (1 + ret).cumprod().cummax() - 1).min()),
        "annual_turnover": float(data.turnover.sum() / len(data) * 252),
        "commission": float(data.commission.sum()),
        "slippage": float(data.slippage.sum()),
        "total_cost": float(data.total_cost.sum()),
    }


def score_series(group, name):
    f = group
    if name == "trend_short_mid":
        return (f.mom_20d + f.mom_60d + f.close_to_ma_60d) / 3
    if name == "trend_mid_slow":
        return (f.mom_60d + f.mom_120d + f.close_to_ma_120d) / 3
    if name == "trend_all":
        return (f.mom_20d + f.mom_60d + f.mom_120d + f.mom_240d) / 4
    if name == "short_reversal":
        return -(f.mom_5d + f.mom_20d) / 2
    if name == "reversal_recovery":
        return -(f.mom_20d) + f.mom_120d
    if name == "volatility_gate":
        vol_median = f.vol_20d.rolling(252, min_periods=60).median()
        return (f.mom_60d > 0).astype(float) - 0.5 + (f.vol_20d <= vol_median).astype(float) * 0.25
    if name == "drawdown_recovery":
        return f.mom_60d + f.drawdown_120d
    if name == "slow_drawdown":
        return f.mom_120d + f.mom_240d + f.drawdown_240d
    raise ValueError(name)


def build_ledger(group, score, start, end, holding, threshold, direction):
    data = group.loc[(group.index >= start) & (group.index <= end)].copy()
    score = score.reindex(data.index) * direction
    target = pd.Series(np.select([score > threshold, score < -threshold], [1.0, 0.0], default=0.5), index=data.index)
    target = target.where(score.notna())
    target = target.where(np.arange(len(target)) % holding == 0).ffill().fillna(0.5)
    data["target_position"] = target
    data["benchmark_return"] = data["next_open_to_open"]
    data["turnover"] = data.target_position.diff().abs().fillna(data.target_position.abs())
    data["commission"] = data.turnover * COMMISSION
    data["slippage"] = data.turnover * SLIPPAGE
    data["total_cost"] = data.commission + data.slippage
    data["gross_strategy_return"] = data.target_position * data.benchmark_return
    data["strategy_return"] = data.gross_strategy_return - data.total_cost
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, choices=["000905", "000852", "399006", "000688", "932000"])
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    profile = profiles[args.index]
    common = profiles["common"]
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    group = panel.loc[panel.index_code.eq(args.index)].sort_values("trade_date").set_index("trade_date")
    contract = pd.read_csv(CONTRACT)
    target_panel = panel.loc[panel.index_code.eq(args.index)]
    factor_frame, _ = screen.build_factor_frame(target_panel, contract, profiles, STATE_QUALITY)
    target_factors = factor_frame.set_index("trade_date").reindex(group.index)
    group = group.join(target_factors[[c for c in target_factors.columns if c not in group.columns]], how="left")
    first = group.index[group.close.notna()].min()
    train_start = group.index[group.index.get_loc(first) + int(profile["oos"]["train_days"]) + max(profile["oos"]["horizons_days"])]
    train_end = pd.Timestamp(profile["oos"]["start"]) - pd.Timedelta(days=1)
    backtest_start, backtest_end = pd.Timestamp(profile["oos"]["start"]), group.index.max()
    strategy_names = ["trend_short_mid", "trend_mid_slow", "trend_all", "short_reversal", "reversal_recovery", "volatility_gate", "drawdown_recovery", "slow_drawdown"]
    thresholds = [0.0, 0.05, 0.10]
    rows, ledgers = [], []
    for name in strategy_names:
        score = score_series(group, name)
        for holding in profile["oos"]["horizons_days"]:
            for threshold in thresholds:
                for direction in (1.0, -1.0):
                    train = build_ledger(group, score, train_start, train_end, holding, threshold, direction)
                    backtest = build_ledger(group, score, backtest_start, backtest_end, holding, threshold, direction)
                    tm, bm = performance(train), performance(backtest)
                    row = {"index_code": args.index, "index_name": profile["name"], "strategy": name, "holding_days": holding, "threshold": threshold, "direction": "direct" if direction > 0 else "inverse", "training_start": train_start.date().isoformat(), "training_end": train_end.date().isoformat(), "backtest_start": backtest_start.date().isoformat(), "backtest_end": backtest_end.date().isoformat(), "training_annualized_excess": tm["annualized_excess"], "training_active_sharpe": tm["active_sharpe"], "backtest_annualized_excess": bm["annualized_excess"], "backtest_active_sharpe": bm["active_sharpe"], "training_rows": tm["rows"], "backtest_rows": bm["rows"], "training_total_cost": tm["total_cost"], "backtest_total_cost": bm["total_cost"]}
                    row["passes_gate"] = bool(row["training_annualized_excess"] > 0 and row["training_active_sharpe"] > 0 and row["backtest_annualized_excess"] > 0 and row["backtest_active_sharpe"] > 0 and row["training_annualized_excess"] >= row["backtest_annualized_excess"])
                    row["gate_score"] = min(row["training_annualized_excess"], row["training_active_sharpe"], row["backtest_annualized_excess"], row["backtest_active_sharpe"])
                    rows.append(row)
                    ledgers.extend([train.assign(period="training", strategy=name, holding_days=holding, threshold=threshold, direction=direction).reset_index(), backtest.assign(period="backtest", strategy=name, holding_days=holding, threshold=threshold, direction=direction).reset_index()])
    out = OUT / args.index
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["passes_gate", "gate_score"], ascending=[False, False])
    result.to_csv(out / "targeted_strategy_results.csv", index=False, encoding="utf-8-sig")
    # Keep only the best candidate ledger; the full grid remains in results.csv.
    best = result.iloc[0] if len(result) else None
    if best is not None:
        best_ledgers = [ledger for ledger in ledgers if ledger["strategy"].eq(best["strategy"]).all() and ledger["holding_days"].eq(best["holding_days"]).all() and ledger["threshold"].eq(best["threshold"]).all() and ledger["direction"].eq(1.0 if best["direction"] == "direct" else -1.0).all()]
        if best_ledgers:
            pd.concat(best_ledgers, ignore_index=True).to_csv(out / "best_strategy_ledgers.csv", index=False, encoding="utf-8-sig")
    summary = {"index": args.index, "index_name": profile["name"], "training_interval": [train_start.date().isoformat(), train_end.date().isoformat()], "backtest_interval": [backtest_start.date().isoformat(), backtest_end.date().isoformat()], "grid_rows": len(result), "passing_strategies": int(result.passes_gate.sum()), "best": result.iloc[0].to_dict(), "commission_one_way": COMMISSION, "slippage_one_way": SLIPPAGE, "total_cost_one_way": TOTAL_COST}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

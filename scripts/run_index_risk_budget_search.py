"""Search low-turnover, risk-budgeted timing rules for failed indices."""

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
OUT = DATA_DIR / "index_risk_budget_search"
COMMISSION = 0.0003
SLIPPAGE = 0.0005
TOTAL_COST = COMMISSION + SLIPPAGE


def metrics(data):
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


def score(group, name):
    f = group
    if name == "trend":
        return (f.mom_20d + f.mom_60d + f.close_to_ma_60d) / 3
    if name == "slow_trend":
        return (f.mom_60d + f.mom_120d + f.close_to_ma_120d + f.drawdown_120d) / 4
    if name == "trend_reversal":
        return (f.mom_120d - f.mom_20d) / 2
    if name == "risk_adjusted_trend":
        return (f.mom_60d + f.close_to_ma_60d) / (1 + f.vol_20d.abs())
    if name == "drawdown_trend":
        return f.mom_60d + f.drawdown_120d + f.mom_120d
    if name == "volatility_reversal":
        return -f.mom_20d + f.mom_60d - f.vol_20d
    raise ValueError(name)


def ledger(group, raw_score, start, end, holding, threshold, mapping, direction):
    data = group.loc[(group.index >= start) & (group.index <= end)].copy()
    s = raw_score.reindex(data.index) * direction
    valid = s.notna()
    if mapping == "binary":
        pos = np.select([s > threshold, s < -threshold], [1.0, 0.0], default=0.5)
    elif mapping == "defensive":
        pos = np.select([s > threshold, s < -threshold], [0.75, 0.25], default=0.5)
    elif mapping == "long_only_defensive":
        pos = np.select([s > threshold, s < -threshold], [1.0, 0.5], default=0.75)
    else:
        raise ValueError(mapping)
    target = pd.Series(pos, index=data.index).where(valid)
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
    parser.add_argument("--index", required=True, choices=["000905", "399006"])
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    profile = profiles[args.index]
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    group = panel.loc[panel.index_code.eq(args.index)].sort_values("trade_date").set_index("trade_date")
    contract = pd.read_csv(CONTRACT)
    target_panel = panel.loc[panel.index_code.eq(args.index)]
    factor_frame, _ = screen.build_factor_frame(target_panel, contract, profiles, STATE_QUALITY)
    extra = factor_frame.set_index("trade_date").reindex(group.index)
    group = group.join(extra[[c for c in extra.columns if c not in group.columns]], how="left")
    first = group.index[group.close.notna()].min()
    train_start = group.index[group.index.get_loc(first) + int(profile["oos"]["train_days"]) + max(profile["oos"]["horizons_days"])]
    train_end = pd.Timestamp(profile["oos"]["start"]) - pd.Timedelta(days=1)
    backtest_start, backtest_end = pd.Timestamp(profile["oos"]["start"]), group.index.max()
    rows = []
    strategy_names = ["trend", "slow_trend", "trend_reversal", "risk_adjusted_trend", "drawdown_trend", "volatility_reversal"]
    for name in strategy_names:
        raw = score(group, name)
        for holding in profile["oos"]["horizons_days"]:
            for threshold in (0.0, 0.05, 0.10, 0.20):
                for mapping in ("binary", "defensive", "long_only_defensive"):
                    for direction in (1.0, -1.0):
                        train = ledger(group, raw, train_start, train_end, holding, threshold, mapping, direction)
                        test = ledger(group, raw, backtest_start, backtest_end, holding, threshold, mapping, direction)
                        tm, bm = metrics(train), metrics(test)
                        row = {"index_code": args.index, "index_name": profile["name"], "strategy": name, "holding_days": holding, "threshold": threshold, "mapping": mapping, "direction": "direct" if direction > 0 else "inverse", "training_start": train_start.date().isoformat(), "training_end": train_end.date().isoformat(), "backtest_start": backtest_start.date().isoformat(), "backtest_end": backtest_end.date().isoformat(), "training_rows": tm["rows"], "backtest_rows": bm["rows"], "training_annualized_excess": tm["annualized_excess"], "training_active_sharpe": tm["active_sharpe"], "backtest_annualized_excess": bm["annualized_excess"], "backtest_active_sharpe": bm["active_sharpe"], "training_annual_turnover": tm["annual_turnover"], "backtest_annual_turnover": bm["annual_turnover"], "training_total_cost": tm["total_cost"], "backtest_total_cost": bm["total_cost"]}
                        row["passes_gate"] = bool(row["training_annualized_excess"] > 0 and row["training_active_sharpe"] > 0 and row["backtest_annualized_excess"] > 0 and row["backtest_active_sharpe"] > 0 and row["training_annualized_excess"] >= row["backtest_annualized_excess"] and row["training_active_sharpe"] >= row["backtest_active_sharpe"])
                        row["gate_score"] = min(row["training_annualized_excess"], row["training_active_sharpe"], row["backtest_annualized_excess"], row["backtest_active_sharpe"])
                        rows.append(row)
    out = OUT / args.index
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["passes_gate", "gate_score"], ascending=[False, False])
    result.to_csv(out / "risk_budget_results.csv", index=False, encoding="utf-8-sig")
    summary = {"index": args.index, "index_name": profile["name"], "training_interval": [train_start.date().isoformat(), train_end.date().isoformat()], "backtest_interval": [backtest_start.date().isoformat(), backtest_end.date().isoformat()], "grid_rows": len(result), "passing_strategies": int(result.passes_gate.sum()), "best": result.iloc[0].to_dict(), "commission_one_way": COMMISSION, "slippage_one_way": SLIPPAGE, "total_cost_one_way": TOTAL_COST}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

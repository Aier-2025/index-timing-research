"""Per-index low-degree model search on train/development and validation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import run_index_oos_worker as worker
import screen_index_timing_factors as screen


DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
CONTRACT = Path(r"F:\量化因子库\docs\reports\csi1000_factor_term_audit_20260819\feature_contract_draft.csv")
STATE_QUALITY = Path(r"F:\量化因子库\docs\factor_library\market_state_factor_quality.csv")
PROFILES = Path(__file__).resolve().parents[1] / "config" / "index_research_profiles.json"
OUT_ROOT = DATA_DIR / "index_model_grid_search"
COMMISSION_ONE_WAY = 0.0003
SLIPPAGE_ONE_WAY = 0.0005
ONE_WAY_COST = COMMISSION_ONE_WAY + SLIPPAGE_ONE_WAY


def position(prediction, cuts, direction, mode):
    bucket = (prediction[:, None] > cuts).sum(axis=1)
    if mode == "long_only":
        levels = np.array([0.50, 0.75, 1.00])
        bucket = np.minimum(bucket, 2)
    else:
        levels = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    values = levels[bucket]
    return values if direction == "direct" else 1.0 - values


def evaluate(period, group, factor_frame, fields, horizon, params, profile):
    dates_all = group.index
    labels = screen.future_return(group.reset_index(), horizon)
    labels.index = group.index
    start = pd.Timestamp(period[0])
    end = pd.Timestamp(period[1])
    dates = dates_all[(dates_all >= start) & (dates_all <= end) & labels.notna()]
    rows = []
    if len(dates) == 0:
        return pd.DataFrame()
    factor_matrix = factor_frame.set_index("trade_date")[fields].reindex(group.index)
    train_days = int(profile["oos"]["train_days"])
    refit_days = int(profile["oos"]["refit_days"])
    for begin in range(0, len(dates), refit_days):
        test_dates = dates[begin:begin + refit_days]
        if len(test_dates) == 0:
            continue
        test_pos = group.index.get_loc(test_dates[0])
        cutoff = max(0, test_pos - horizon)
        completed = labels.iloc[:cutoff].dropna().index
        train_dates = completed[-train_days:]
        if len(train_dates) < train_days:
            continue
        train_x = factor_matrix.reindex(train_dates)
        train_y = labels.reindex(train_dates)
        valid_fields = train_x.notna().sum().ge(max(60, int(train_days * 0.5))) & train_x.nunique(dropna=True).gt(1)
        train_x = train_x.loc[:, valid_fields]
        if train_x.empty:
            continue
        rank_ic = train_x.rank().corrwith(train_y.rank()).abs().sort_values(ascending=False)
        selected = rank_ic.head(int(params["top_k"])).index.tolist()
        train_x, test_x = worker.prepare(train_x[selected], factor_matrix.reindex(test_dates)[selected])
        model = Ridge(alpha=float(params["alpha"])).fit(train_x, train_y.to_numpy(float))
        train_prediction = model.predict(train_x)
        test_prediction = model.predict(test_x)
        cut_count = 2 if params["mode"] == "long_only" else 4
        cuts = np.quantile(train_prediction, np.linspace(0.2, 0.8, cut_count))
        desired = pd.Series(position(test_prediction, cuts, params["direction"], params["mode"]), index=test_dates)
        desired = desired.where(np.arange(len(desired)) % horizon == 0).ffill().fillna(0.5)
        for date, target in desired.items():
            rows.append({"trade_date": date, "target_position": float(target), "horizon_days": horizon, "period": period[2]})
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows).drop_duplicates("trade_date").set_index("trade_date").sort_index()
    data["benchmark_return"] = group["next_open_to_open"].reindex(data.index)
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
    common = profiles["common"]
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    contract = pd.read_csv(CONTRACT)
    target_panel = panel.loc[panel["index_code"].eq(args.index)].copy()
    factor_frame, _ = screen.build_factor_frame(target_panel, contract, profiles, STATE_QUALITY)
    registry = screen.candidate_registry(contract, factor_frame, profile, common)
    fields_by_horizon = {h: [item["field"] for item in registry if h in item["allowed_horizons"]] for h in profile["oos"]["horizons_days"]}
    group = target_panel.sort_values("trade_date").set_index("trade_date")
    first_valid = group.index[group["close"].notna()].min()
    train_start = group.index[group.index.get_loc(first_valid) + int(profile["oos"]["train_days"]) + max(profile["oos"]["horizons_days"])]
    train_end = pd.Timestamp(profile["oos"]["start"]) - pd.Timedelta(days=1)
    valid_start = pd.Timestamp(profile["oos"]["start"])
    valid_end = group.index.max()
    params_grid = [{"alpha": alpha, "top_k": top_k, "direction": direction, "mode": mode}
                   for alpha in (1.0, 20.0, 100.0)
                   for top_k in (5, 10, 15)
                   for direction in ("direct", "inverse")
                   for mode in ("five_level", "long_only")]
    rows = []
    ledgers = []
    for horizon in profile["oos"]["horizons_days"]:
        fields = fields_by_horizon[horizon]
        for params in params_grid:
            train = evaluate((train_start, train_end, "training"), group, factor_frame, fields, horizon, params, profile)
            valid = evaluate((valid_start, valid_end, "validation"), group, factor_frame, fields, horizon, params, profile)
            train_m = worker.metrics(train)
            valid_m = worker.metrics(valid)
            if not train_m.get("rows") or not valid_m.get("rows"):
                continue
            row = {"index_code": args.index, "index_name": profile["name"], "horizon_days": horizon, **params,
                   "training_start": str(train_start.date()), "training_end": str(train_end.date()), "backtest_start": str(valid_start.date()), "backtest_end": str(valid_end.date()),
                   "training_rows": train_m["rows"], "training_annualized_excess": train_m["annualized_excess"], "training_active_sharpe": train_m["active_sharpe"],
                   "backtest_rows": valid_m["rows"], "backtest_annualized_excess": valid_m["annualized_excess"], "backtest_active_sharpe": valid_m["active_sharpe"]}
            row["passes_gate"] = bool(row["training_annualized_excess"] > 0 and row["training_active_sharpe"] > 0 and row["backtest_annualized_excess"] > 0 and row["backtest_active_sharpe"] > 0 and row["training_annualized_excess"] >= row["backtest_annualized_excess"] and row["training_active_sharpe"] >= row["backtest_active_sharpe"])
            row["gate_score"] = float(min(row["training_annualized_excess"], row["training_active_sharpe"], row["backtest_annualized_excess"], row["backtest_active_sharpe"]))
            rows.append(row)
            ledgers.extend([train.assign(strategy_period="training", horizon_days=horizon, **params).reset_index(), valid.assign(strategy_period="backtest", horizon_days=horizon, **params).reset_index()])
    out = OUT_ROOT / args.index
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["passes_gate", "gate_score"], ascending=[False, False])
    result.to_csv(out / "model_grid_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(ledgers, ignore_index=True).to_csv(out / "model_grid_ledgers.csv", index=False, encoding="utf-8-sig")
    summary = {"index": args.index, "index_name": profile["name"], "training_interval": [str(train_start.date()), str(train_end.date())], "backtest_interval": [str(valid_start.date()), str(valid_end.date())], "grid_rows": int(len(result)), "passing_strategies": int(result["passes_gate"].sum()) if not result.empty else 0, "best": result.iloc[0].to_dict() if not result.empty else None, "backtest_status": "not_run"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

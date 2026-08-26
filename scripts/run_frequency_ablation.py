"""Run pre-declared frequency ablations for the four effective index candidates.

The runner reuses the project's point-in-time factor construction and rolling
training contract. It writes a new result tree and never changes historical
OOS worker outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_index_oos_worker as worker
import screen_index_timing_factors as screen


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
CONTRACT = Path(r"F:\量化因子库\docs\reports\csi1000_factor_term_audit_20260819\feature_contract_draft.csv")
STATE_QUALITY = Path(r"F:\量化因子库\docs\factor_library\market_state_factor_quality.csv")
PROFILES = ROOT / "config" / "index_research_profiles.json"
OUT_ROOT = DATA_DIR / "index_frequency_ablation_20260826"

COMMISSION = 0.0003
SLIPPAGE = 0.0005

INDEX_CONFIG = {
    "000905": {"horizon": 20, "train_days": 756, "top_k": 20, "oos_start": "2019-01-01"},
    "000852": {"horizon": 10, "train_days": 504, "top_k": 15, "oos_start": "2021-01-01"},
    "399006": {"horizon": 20, "train_days": 504, "top_k": 15, "oos_start": "2021-01-01"},
    "000688": {"horizon": 10, "train_days": 252, "top_k": 10, "oos_start": "2023-01-01"},
}

MODES = {
    "technical_only": "独立 OHLC 技术因子",
    "fast_55_full": "55 个响应型日频合同因子",
    "fast_plus_medium": "快因子 + 中频因子直接拼接",
    "fast_plus_slow_raw": "快因子 + 慢因子原始水平直接拼接",
    "fast_plus_slow_transformed": "快因子 + 慢因子变化/偏离变换",
    "all_available": "技术 + 快 + 中 + 慢全部输入",
    "fast_slow_blend": "快专家与慢专家双模型预测融合",
}


def empty_metrics() -> dict:
    return {
        "rows": 0,
        "annualized_return": np.nan,
        "benchmark_annualized_return": np.nan,
        "annualized_excess": np.nan,
        "active_sharpe": np.nan,
        "max_drawdown": np.nan,
        "annual_turnover": np.nan,
        "total_cost": np.nan,
    }


def factor_source_label(source: str, field: str) -> str:
    if source == "alpha101" or field.startswith("alpha_"):
        return "Alpha101"
    if source == "macro_csv_2015_plus":
        return "股票宏观因子库"
    if source == "stock_pool_2000_derived":
        return "市场情绪因子/股票池聚合派生"
    if source == "market_state_factor_panel":
        return "组合衍生因子/市场状态"
    if source == "derived_transform":
        return "组合衍生因子/慢变量变换"
    if source == "local_index_cache":
        return "独立因子/跨指数相对强弱"
    if source == "project_ohlc":
        return "独立因子/指数 OHLC"
    if source == "public_macro_2005_plus":
        return "独立因子/公开宏观"
    return source or "未映射来源"


def mode_group(horizon_bucket: str) -> str:
    if horizon_bucket == "responsive_daily":
        return "fast"
    if horizon_bucket == "medium_daily":
        return "medium"
    if horizon_bucket in {"persistent_daily", "lagged_persistent", "slow_release"}:
        return "slow"
    return "other"


def build_registry(factor_frame: pd.DataFrame, contract: pd.DataFrame, profile: dict, common: dict) -> pd.DataFrame:
    rows = []
    contract_map = contract.set_index("field")
    technical = [
        c for c in factor_frame.columns
        if c.startswith(("mom_", "close_to_ma_", "vol_", "drawdown_", "intraday_", "close_location"))
    ]
    for field in technical:
        rows.append({
            "field": field, "source": "project_ohlc", "family": "price_technical",
            "horizon_bucket": "responsive_daily", "group": "technical", "allowed": True,
            "source_label": factor_source_label("project_ohlc", field), "base_field": field,
            "transform": "level",
        })
    for field in factor_frame.columns:
        if field in {"trade_date", *technical}:
            continue
        if field in contract_map.index:
            item = contract_map.loc[field]
            allowed = set(screen.parse_horizons(item.allowed_research_horizons_days))
            if profile.get("horizon") is not None and int(profile["horizon"]) not in allowed:
                continue
            source = str(item.source)
            bucket = str(item.horizon_bucket)
            rows.append({
                "field": field, "source": source, "family": str(item.family),
                "horizon_bucket": bucket, "group": mode_group(bucket), "allowed": True,
                "source_label": factor_source_label(source, field), "base_field": field,
                "transform": "level",
            })
        elif field.startswith(("breadth.", "flow.", "industry.", "valuation.", "basis.")):
            rows.append({
                "field": field, "source": "market_state_factor_panel", "family": "market_state",
                "horizon_bucket": "medium_daily", "group": "medium", "allowed": True,
                "source_label": factor_source_label("market_state_factor_panel", field), "base_field": field,
                "transform": "level",
            })
    return pd.DataFrame(rows).drop_duplicates("field")


def add_slow_transforms(frame: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    additions = {}
    rows = []
    slow = registry.loc[registry["group"].eq("slow")]
    for item in slow.itertuples(index=False):
        if item.field not in frame.columns:
            continue
        series = pd.to_numeric(frame[item.field], errors="coerce")
        transforms = {
            "chg20": series.diff(20),
            "chg60": series.diff(60),
            "gap60": series - series.rolling(60, min_periods=30).mean(),
            "pct120": series.rolling(120, min_periods=60).rank(pct=True),
        }
        for suffix, values in transforms.items():
            name = f"{item.field}__{suffix}"
            additions[name] = values
            rows.append({
                "field": name, "source": "derived_transform", "family": item.family,
                "horizon_bucket": item.horizon_bucket, "group": "slow_transform",
                "source_label": factor_source_label("derived_transform", name),
                "base_field": item.field, "transform": suffix,
            })
    additions_frame = pd.DataFrame(additions, index=frame.index)
    output = pd.concat([frame, additions_frame], axis=1) if not additions_frame.empty else frame.copy()
    catalog_additions = pd.DataFrame(rows)
    return output, pd.concat([registry, catalog_additions], ignore_index=True) if not catalog_additions.empty else registry


def fields_for_mode(registry: pd.DataFrame, mode: str) -> list[str]:
    if mode == "technical_only":
        groups = {"technical"}
    elif mode == "fast_55_full":
        groups = {"fast"}
    elif mode == "fast_plus_medium":
        groups = {"fast", "medium"}
    elif mode == "fast_plus_slow_raw":
        groups = {"fast", "slow"}
    elif mode == "fast_plus_slow_transformed":
        groups = {"fast", "slow_transform"}
    elif mode == "all_available":
        groups = {"technical", "fast", "medium", "slow", "slow_transform"}
    elif mode == "fast_slow_blend":
        groups = {"fast", "slow"}
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return registry.loc[registry["group"].isin(groups), "field"].drop_duplicates().tolist()


def select_fields(train_x: pd.DataFrame, train_y: pd.Series, fields: list[str], top_k: int, full_fast: bool) -> list[str]:
    available = [c for c in fields if c in train_x.columns]
    if not available:
        return []
    x = train_x[available]
    valid = x.notna().sum().ge(max(60, int(len(train_x) * 0.5))) & x.nunique(dropna=True).gt(1)
    x = x.loc[:, valid]
    if x.empty:
        return []
    if full_fast:
        return x.columns.tolist()
    scores = x.rank().corrwith(train_y.rank()).abs().replace([np.inf, -np.inf], np.nan).dropna()
    return scores.sort_values(ascending=False).head(top_k).index.tolist()


def fit_ridge(train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series, alpha: float = 1.0):
    train_x, test_x = worker.prepare(train_x, test_x)
    model = Ridge(alpha=alpha)
    model.fit(train_x, train_y.to_numpy(float))
    return pd.Series(model.predict(train_x), index=train_x.index), pd.Series(model.predict(test_x), index=test_x.index)


def position_from_prediction(train_prediction: pd.Series, test_prediction: pd.Series, direction: str, horizon: int) -> pd.Series:
    cuts = train_prediction.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    bucket = np.select(
        [test_prediction <= cuts[0], test_prediction <= cuts[1], test_prediction <= cuts[2], test_prediction <= cuts[3]],
        [0.0, 0.25, 0.5, 0.75], default=1.0,
    )
    if direction == "inverse":
        bucket = 1.0 - bucket
    position = pd.Series(bucket, index=test_prediction.index)
    return position.where(np.arange(len(position)) % horizon == 0).ffill().fillna(0.5)


def make_ledger(group: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    data = signal.join(group[["next_open_to_open"]], how="left").rename(columns={"next_open_to_open": "benchmark_return"})
    data["turnover"] = data["target_position"].diff().abs().fillna(data["target_position"].abs())
    data["commission"] = data["turnover"] * COMMISSION
    data["slippage"] = data["turnover"] * SLIPPAGE
    data["total_cost"] = data["commission"] + data["slippage"]
    data["gross_strategy_return"] = data["target_position"] * data["benchmark_return"]
    data["strategy_return"] = data["gross_strategy_return"] - data["total_cost"]
    return data


def run_period(group: pd.DataFrame, factor_frame: pd.DataFrame, registry: pd.DataFrame, mode: str,
               direction: str, horizon: int, train_days: int, start: pd.Timestamp, end: pd.Timestamp,
               top_k: int, period_name: str) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    labels = screen.future_return(group.reset_index(), horizon)
    labels.index = group.index
    dates = group.index[(group.index >= start) & (group.index <= end) & labels.notna()]
    if dates.empty:
        return pd.DataFrame(), [], []
    field_list = fields_for_mode(registry, mode)
    matrix = factor_frame.set_index("trade_date")[field_list].reindex(group.index)
    rows, folds, selected_rows = [], [], []
    refit_days = 63
    for begin in range(0, len(dates), refit_days):
        test_dates = dates[begin: begin + refit_days]
        first_test_pos = group.index.get_loc(test_dates[0])
        cutoff_pos = max(0, first_test_pos - horizon)
        completed = labels.iloc[:cutoff_pos].dropna().index
        train_dates = completed[-train_days:]
        if len(train_dates) < train_days:
            continue
        train_x = matrix.reindex(train_dates)
        test_x = matrix.reindex(test_dates)
        train_y = labels.reindex(train_dates)
        if mode == "fast_slow_blend":
            fast_fields = fields_for_mode(registry, "fast_55_full")
            slow_fields = fields_for_mode(registry, "fast_plus_slow_raw")
            slow_fields = [c for c in slow_fields if c not in fast_fields]
            fast_selected = select_fields(train_x, train_y, fast_fields, top_k, False)
            slow_selected = select_fields(train_x, train_y, slow_fields, top_k, False)
            if not fast_selected or not slow_selected:
                continue
            fast_train, fast_test = fit_ridge(train_x[fast_selected], test_x[fast_selected], train_y)
            slow_train, slow_test = fit_ridge(train_x[slow_selected], test_x[slow_selected], train_y)
            fast_scale = fast_train.std(ddof=0) or 1.0
            slow_scale = slow_train.std(ddof=0) or 1.0
            train_prediction = 0.5 * ((fast_train - fast_train.mean()) / fast_scale) + 0.5 * ((slow_train - slow_train.mean()) / slow_scale)
            test_prediction = 0.5 * ((fast_test - fast_train.mean()) / fast_scale) + 0.5 * ((slow_test - slow_train.mean()) / slow_scale)
            selected = fast_selected + slow_selected
            selected_groups = {"fast": fast_selected, "slow": slow_selected}
        else:
            selected = select_fields(train_x, train_y, field_list, top_k, mode == "fast_55_full")
            if not selected:
                continue
            train_prediction, test_prediction = fit_ridge(train_x[selected], test_x[selected], train_y)
            selected_groups = {}
            for item in registry.loc[registry.field.isin(selected)].itertuples(index=False):
                selected_groups.setdefault(item.group, []).append(item.field)
        position = position_from_prediction(train_prediction, test_prediction, direction, horizon)
        for day in test_dates:
            rows.append({
                "trade_date": day, "period": period_name, "mode": mode, "direction": direction,
                "horizon_days": horizon, "target_position": float(position.loc[day]),
                "prediction": float(test_prediction.loc[day]),
                "selected_features": ";".join(selected),
            })
        folds.append({
            "period": period_name, "mode": mode, "direction": direction, "horizon_days": horizon,
            "test_start": test_dates[0], "test_end": test_dates[-1],
            "train_start": train_dates[0], "train_end": train_dates[-1],
            "train_rows": len(train_dates), "selected_feature_count": len(selected),
            "selected_features": ";".join(selected),
        })
        for group_name, names in selected_groups.items():
            for name in names:
                selected_rows.append({
                    "period": period_name, "mode": mode, "direction": direction,
                    "test_start": test_dates[0], "test_end": test_dates[-1],
                    "group": group_name, "field": name,
                })
    if not rows:
        return pd.DataFrame(), folds, selected_rows
    signal = pd.DataFrame(rows).drop_duplicates("trade_date").set_index("trade_date").sort_index()
    return make_ledger(group, signal), folds, selected_rows


def baseline_rule(group: pd.DataFrame, code: str, start: pd.Timestamp, end: pd.Timestamp, horizon: int) -> pd.DataFrame:
    if code == "000905":
        score = -group["mom_20d"] + group["mom_60d"] - group["vol_20d"]
        direction, threshold = -1.0, 0.10
    elif code == "399006":
        score = -group["mom_20d"] + group["mom_60d"] - group["vol_20d"]
        direction, threshold = -1.0, 0.05
    else:
        return pd.DataFrame()
    data = group.loc[(group.index >= start) & (group.index <= end)].copy()
    score = score.reindex(data.index) * direction
    target = pd.Series(np.select([score > threshold, score < -threshold], [1.0, 0.0], default=0.5), index=data.index)
    target = target.where(np.arange(len(target)) % horizon == 0).ffill().fillna(0.5)
    return make_ledger(data, pd.DataFrame({"target_position": target}, index=data.index))


def source_summary(catalog: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    counts = selected.groupby(["mode", "direction", "field"]).size().rename("selected_fold_count").reset_index()
    result = counts.merge(catalog[["field", "source", "source_label", "family", "group", "transform", "base_field"]], on="field", how="left")
    return result.groupby(["mode", "direction", "source_label", "source", "group", "family"], dropna=False).agg(
        selected_fold_count=("selected_fold_count", "sum"), unique_fields=("field", "nunique"),
    ).reset_index().sort_values(["mode", "selected_fold_count"], ascending=[True, False])


def run_index(code: str, panel: pd.DataFrame, contract: pd.DataFrame, profiles: dict, output_root: Path) -> dict:
    profile = profiles[code]
    cfg = INDEX_CONFIG[code]
    target = panel.loc[panel["index_code"].eq(code)].copy()
    factor_frame, state_lags = screen.build_factor_frame(target, contract, profiles, STATE_QUALITY)
    registry = build_registry(factor_frame, contract, {"horizon": cfg["horizon"]}, profiles["common"])
    factor_frame, registry = add_slow_transforms(factor_frame, registry)
    group = target.sort_values("trade_date").set_index("trade_date")
    first_valid = group.index[group["close"].notna()].min()
    development_start = group.index[group.index.get_loc(first_valid) + cfg["train_days"] + cfg["horizon"]]
    development_end = pd.Timestamp(cfg["oos_start"]) - pd.Timedelta(days=1)
    replay_start = pd.Timestamp(cfg["oos_start"])
    replay_end = group.index.max()
    rows, ledgers, folds, selected = [], [], [], []
    for mode in MODES:
        directions = ("direct", "inverse")
        for direction in directions:
            train, train_folds, train_selected = run_period(group, factor_frame, registry, mode, direction, cfg["horizon"], cfg["train_days"], development_start, development_end, cfg["top_k"], "development")
            replay, replay_folds, replay_selected = run_period(group, factor_frame, registry, mode, direction, cfg["horizon"], cfg["train_days"], replay_start, replay_end, cfg["top_k"], "historical_replay")
            train_metrics = worker.metrics(train) if {"strategy_return", "benchmark_return"}.issubset(train.columns) else empty_metrics()
            replay_metrics = worker.metrics(replay) if {"strategy_return", "benchmark_return"}.issubset(replay.columns) else empty_metrics()
            row = {
                "index_code": code, "index_name": profile["name"], "mode": mode,
                "mode_label": MODES[mode], "direction": direction, "horizon_days": cfg["horizon"],
                "development_start": development_start.date().isoformat(), "development_end": development_end.date().isoformat(),
                "replay_start": replay_start.date().isoformat(), "replay_end": replay_end.date().isoformat(),
                "development_rows": train_metrics.get("rows", 0), "replay_rows": replay_metrics.get("rows", 0),
            }
            for prefix, metrics in (("development", train_metrics), ("replay", replay_metrics)):
                for key in ("annualized_return", "benchmark_annualized_return", "annualized_excess", "active_sharpe", "max_drawdown", "annual_turnover", "total_cost"):
                    row[f"{prefix}_{key}"] = metrics.get(key)
            row["development_score"] = float(min(train_metrics.get("annualized_excess", np.nan), train_metrics.get("active_sharpe", np.nan)))
            row["replay_score"] = float(min(replay_metrics.get("annualized_excess", np.nan), replay_metrics.get("active_sharpe", np.nan)))
            row["development_pass"] = bool(train_metrics.get("annualized_excess", -1) > 0 and train_metrics.get("active_sharpe", -1) > 0)
            row["replay_pass"] = bool(replay_metrics.get("annualized_excess", -1) > 0 and replay_metrics.get("active_sharpe", -1) > 0)
            rows.append(row)
            if not train.empty:
                ledgers.append(train.reset_index())
            if not replay.empty:
                ledgers.append(replay.reset_index())
            folds.extend(train_folds + replay_folds)
            selected.extend(train_selected + replay_selected)
    out = output_root / code
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "ablation_results.csv", index=False, encoding="utf-8-sig")
    if ledgers:
        pd.concat(ledgers, ignore_index=True).to_csv(out / "ablation_ledgers.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folds).to_csv(out / "fold_audit.csv", index=False, encoding="utf-8-sig")
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(out / "selected_features_by_fold.csv", index=False, encoding="utf-8-sig")
    catalog = registry.copy()
    catalog.to_csv(out / "factor_catalog.csv", index=False, encoding="utf-8-sig")
    source_summary(catalog, selected_frame).to_csv(out / "source_contribution.csv", index=False, encoding="utf-8-sig")
    best = result.sort_values(["development_pass", "development_score", "development_active_sharpe"], ascending=[False, False, False]).iloc[0].to_dict() if not result.empty else None
    summary = {
        "index": code, "index_name": profile["name"], "horizon_days": cfg["horizon"],
        "development_interval": [development_start.date().isoformat(), development_end.date().isoformat()],
        "historical_replay_interval": [replay_start.date().isoformat(), replay_end.date().isoformat()],
        "maturity": "historical_replay_not_new_oos",
        "best_by_development": best,
        "modes": MODES, "factor_count": int(len(catalog)),
        "factor_group_counts": catalog.groupby("group").size().to_dict(),
        "state_lags": state_lags,
        "cost": {"commission_one_way": COMMISSION, "slippage_one_way": SLIPPAGE, "total_one_way": COMMISSION + SLIPPAGE},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=list(INDEX_CONFIG))
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    contract = pd.read_csv(CONTRACT)
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    codes = [args.index] if args.index else list(INDEX_CONFIG)
    summaries = [run_index(code, panel, contract, profiles, args.output_root) for code in codes]
    print(json.dumps({"output_root": str(args.output_root), "indices": summaries}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

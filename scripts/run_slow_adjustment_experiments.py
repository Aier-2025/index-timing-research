"""Test independent slow-factor adjustments on top of a fast-factor model.

Fast and slow factors are never concatenated into one estimator. The fast model
sets the base position; an independently trained slow model either caps,
scales, gates, or residual-adjusts that position.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_index_oos_worker as worker
import screen_index_timing_factors as screen
from run_frequency_ablation import (
    add_slow_transforms,
    build_registry,
    empty_metrics,
    factor_source_label,
    select_fields,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
CONTRACT = Path(r"F:\量化因子库\docs\reports\csi1000_factor_term_audit_20260819\feature_contract_draft.csv")
STATE_QUALITY = Path(r"F:\量化因子库\docs\factor_library\market_state_factor_quality.csv")
PROFILES = ROOT / "config" / "index_research_profiles.json"
OUT_ROOT = DATA_DIR / "index_slow_adjustment_20260826"

COMMISSION = 0.0003
SLIPPAGE = 0.0005

SCHEMES = {
    "parallel_cap": "并行慢模型分位数仓位上限",
    "parallel_multiplier": "并行慢模型连续仓位缩放",
    "parallel_gate": "并行慢模型状态门控",
    "serial_residual_cap": "串行残差模型仓位上限",
    "serial_residual_multiplier": "串行残差模型连续仓位缩放",
}

SLOW_CONFIG = {
    "000905": {"fast_horizons": [10, 20], "slow_horizons": [20, 60], "fold_days": [63, 126], "directions": ["inverse"], "train_days": 756, "top_k": 20, "oos_start": "2019-01-01"},
    "000852": {"fast_horizons": [5, 10], "slow_horizons": [20, 60], "fold_days": [42, 63, 126], "directions": ["direct"], "train_days": 504, "top_k": 15, "oos_start": "2021-01-01"},
    "399006": {"fast_horizons": [5, 10], "slow_horizons": [20, 60], "fold_days": [42, 63, 126], "directions": ["inverse"], "train_days": 504, "top_k": 15, "oos_start": "2021-01-01"},
    "000688": {"fast_horizons": [5, 10], "slow_horizons": [20, 60], "fold_days": [42, 63], "directions": ["inverse"], "train_days": 252, "top_k": 10, "oos_start": "2023-01-01"},
}


def make_factor_catalog(factor_frame: pd.DataFrame, contract: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = build_registry(factor_frame, contract, {"horizon": None}, profile["common"])
    factor_frame, registry = add_slow_transforms(factor_frame, registry)
    # The main fast model includes project OHLC timing features and the 55
    # responsive contract fields, while the slow model receives only slow fields.
    registry.loc[registry["group"].eq("technical"), "model_role"] = "fast"
    registry.loc[registry["group"].eq("fast"), "model_role"] = "fast"
    registry.loc[registry["group"].isin(["slow", "slow_transform"]), "model_role"] = "slow"
    registry["model_role"] = registry["model_role"].fillna("excluded")
    registry["source_label"] = registry.apply(
        lambda row: factor_source_label(row["source"], row["field"]), axis=1
    )
    return factor_frame, registry


def fields(registry: pd.DataFrame, role: str) -> list[str]:
    return registry.loc[registry["model_role"].eq(role), "field"].drop_duplicates().tolist()


def fit_model(train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series, alpha: float = 20.0):
    train_x, test_x = worker.prepare(train_x, test_x)
    model = Ridge(alpha=alpha)
    model.fit(train_x, train_y.to_numpy(float))
    return (
        pd.Series(model.predict(train_x), index=train_x.index),
        pd.Series(model.predict(test_x), index=test_x.index),
    )


def fit_fast(train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series, top_k: int):
    selected = select_fields(train_x, train_y, list(train_x.columns), top_k, False)
    if not selected:
        return None, None, []
    train_prediction, test_prediction = fit_model(train_x[selected], test_x[selected], train_y)
    return train_prediction, test_prediction, selected


def cross_fitted_fast_predictions(train_x: pd.DataFrame, train_y: pd.Series, top_k: int) -> pd.Series:
    """Create residual-training predictions without using same-row fitted values."""
    result = pd.Series(np.nan, index=train_x.index, dtype=float)
    n = len(train_x)
    if n < 120:
        return result
    block = max(40, n // 4)
    for begin in range(block, n, block):
        end = min(begin + block, n)
        prior_x = train_x.iloc[:begin]
        prior_y = train_y.iloc[:begin]
        selected = select_fields(prior_x, prior_y, list(prior_x.columns), top_k, False)
        if not selected:
            continue
        _, prediction = fit_model(prior_x[selected], train_x.iloc[begin:end][selected], prior_y)
        result.iloc[begin:end] = prediction.to_numpy(float)
    return result


def quantile_bucket(values: pd.Series, reference: pd.Series) -> pd.Series:
    q = reference.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    return pd.Series(
        np.select(
            [values <= q[0], values <= q[1], values <= q[2], values <= q[3]],
            [0.0, 0.25, 0.5, 0.75],
            default=1.0,
        ),
        index=values.index,
    )


def adjust_position(base: pd.Series, slow_train: pd.Series, slow_test: pd.Series, scheme: str) -> pd.Series:
    slow_rank = slow_test.rank(pct=True)
    if scheme.endswith("cap"):
        # Slow model controls only the maximum fast-model exposure.
        cap = pd.Series(
            np.select(
                [slow_test <= slow_train.quantile(0.2), slow_test <= slow_train.quantile(0.4),
                 slow_test <= slow_train.quantile(0.6), slow_test <= slow_train.quantile(0.8)],
                [0.25, 0.50, 0.75, 1.00],
                default=1.00,
            ),
            index=slow_test.index,
        )
        return base.clip(lower=0.0, upper=cap)
    if scheme.endswith("multiplier"):
        # No leverage: a weak slow forecast can reduce, but never reverse,
        # the fast model's direction or increase exposure above 100%.
        multiplier = (0.35 + 0.65 * slow_rank).clip(0.35, 1.0)
        return (base * multiplier).clip(0.0, 1.0)
    if scheme.endswith("gate"):
        median = slow_train.median()
        return base.where(slow_test >= median, base.clip(upper=0.50))
    raise ValueError(f"unsupported adjustment scheme: {scheme}")


def apply_direction(prediction: pd.Series, train_prediction: pd.Series, direction: str, horizon: int) -> pd.Series:
    base = quantile_bucket(prediction, train_prediction)
    if direction == "inverse":
        base = 1.0 - base
    return base.where(np.arange(len(base)) % horizon == 0).ffill().fillna(0.5)


def make_ledger(group: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"target_position": position}, index=position.index).join(
        group[["next_open_to_open"]], how="left"
    ).rename(columns={"next_open_to_open": "benchmark_return"})
    data["turnover"] = data["target_position"].diff().abs().fillna(data["target_position"].abs())
    data["commission"] = data["turnover"] * COMMISSION
    data["slippage"] = data["turnover"] * SLIPPAGE
    data["total_cost"] = data["commission"] + data["slippage"]
    data["gross_strategy_return"] = data["target_position"] * data["benchmark_return"]
    data["strategy_return"] = data["gross_strategy_return"] - data["total_cost"]
    return data


def run_fold(
    group: pd.DataFrame,
    matrix: pd.DataFrame,
    mode: str,
    direction: str,
    fast_horizon: int,
    slow_horizon: int,
    fold_days: int,
    train_days: int,
    top_k: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period: str,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    fast_labels = screen.future_return(group.reset_index(), fast_horizon)
    fast_labels.index = group.index
    slow_labels = screen.future_return(group.reset_index(), slow_horizon)
    slow_labels.index = group.index
    dates = group.index[(group.index >= start) & (group.index <= end) & fast_labels.notna()]
    if dates.empty:
        return pd.DataFrame(), [], []

    fast_fields = [c for c in matrix.columns if c.startswith("fast::")]
    slow_fields = [c for c in matrix.columns if c.startswith("slow::")]
    fast_matrix = matrix[fast_fields].rename(columns=lambda c: c.removeprefix("fast::"))
    slow_matrix = matrix[slow_fields].rename(columns=lambda c: c.removeprefix("slow::"))
    rows, fold_rows, selection_rows = [], [], []

    for begin in range(0, len(dates), fold_days):
        test_dates = dates[begin:begin + fold_days]
        first_test_pos = group.index.get_loc(test_dates[0])
        fast_cutoff_pos = max(0, first_test_pos - fast_horizon)
        slow_cutoff_pos = max(0, first_test_pos - slow_horizon)
        fast_completed = fast_labels.iloc[:fast_cutoff_pos].dropna().index
        slow_completed = slow_labels.iloc[:slow_cutoff_pos].dropna().index
        fast_train_dates = fast_completed[-train_days:]
        slow_train_dates = slow_completed[-train_days:]
        if len(fast_train_dates) < train_days or len(slow_train_dates) < train_days:
            continue
        fast_train_y = fast_labels.reindex(fast_train_dates)
        slow_train_y = slow_labels.reindex(slow_train_dates)
        fast_train_x = fast_matrix.reindex(fast_train_dates)
        fast_test_x = fast_matrix.reindex(test_dates)
        slow_train_x = slow_matrix.reindex(slow_train_dates)
        slow_test_x = slow_matrix.reindex(test_dates)

        if mode.startswith("parallel"):
            fast_train_pred, fast_test_pred, fast_selected = fit_fast(fast_train_x, fast_test_x, fast_train_y, top_k)
            slow_selected = select_fields(slow_train_x, slow_train_y, list(slow_train_x.columns), top_k, False)
            if fast_train_pred is None or not slow_selected:
                continue
            slow_train_pred, slow_test_pred = fit_model(
                slow_train_x[slow_selected], slow_test_x[slow_selected], slow_train_y
            )
        elif mode.startswith("serial_residual"):
            fast_train_pred, fast_test_pred, fast_selected = fit_fast(fast_train_x, fast_test_x, fast_train_y, top_k)
            if fast_train_pred is None:
                continue
            # Serial mode uses the slow model only for the fast-horizon
            # residual, but enforces the slower maturity buffer above.
            cross_pred = cross_fitted_fast_predictions(fast_train_x[fast_selected], fast_train_y, top_k)
            residual = (fast_train_y - cross_pred).dropna()
            if len(residual) < max(80, train_days // 3):
                continue
            slow_selected = select_fields(
                slow_matrix.reindex(residual.index), residual, list(slow_matrix.columns), top_k, False
            )
            if not slow_selected:
                continue
            slow_train_pred, slow_test_pred = fit_model(
                slow_matrix.reindex(residual.index)[slow_selected],
                slow_test_x[slow_selected],
                residual,
            )
        else:
            raise ValueError(mode)

        base_position = apply_direction(fast_test_pred, fast_train_pred, direction, fast_horizon)
        adjusted = adjust_position(base_position, slow_train_pred, slow_test_pred, mode)
        for day in test_dates:
            rows.append({
                "trade_date": day,
                "period": period,
                "scheme": mode,
                "direction": direction,
                "fast_horizon_days": fast_horizon,
                "slow_horizon_days": slow_horizon,
                "fold_days": fold_days,
                "fast_prediction": float(fast_test_pred.loc[day]),
                "slow_prediction": float(slow_test_pred.loc[day]),
                "base_position": float(base_position.loc[day]),
                "target_position": float(adjusted.loc[day]),
            })
        fold_rows.append({
            "period": period,
            "scheme": mode,
            "direction": direction,
            "fast_horizon_days": fast_horizon,
            "slow_horizon_days": slow_horizon,
            "fold_days": fold_days,
            "fast_train_start": fast_train_dates[0],
            "fast_train_end": fast_train_dates[-1],
            "slow_train_start": slow_train_dates[0],
            "slow_train_end": slow_train_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "train_rows": len(train_dates),
            "fast_feature_count": len(fast_selected),
            "slow_feature_count": len(slow_selected),
            "fast_features": ";".join(fast_selected),
            "slow_features": ";".join(slow_selected),
        })
        for role, selected in (("fast", fast_selected), ("slow", slow_selected)):
            for field in selected:
                selection_rows.append({
                    "period": period, "scheme": mode, "direction": direction,
                    "fast_horizon_days": fast_horizon, "slow_horizon_days": slow_horizon,
                    "fold_days": fold_days,
                    "test_start": test_dates[0], "test_end": test_dates[-1],
                    "role": role, "field": field,
                })
    if not rows:
        return pd.DataFrame(), fold_rows, selection_rows
    signal = pd.DataFrame(rows).drop_duplicates("trade_date").set_index("trade_date").sort_index()
    return make_ledger(group, signal["target_position"]), fold_rows, selection_rows


def run_index(code: str, panel: pd.DataFrame, contract: pd.DataFrame, profiles: dict, output_root: Path) -> dict:
    profile = profiles[code]
    cfg = SLOW_CONFIG[code]
    target = panel.loc[panel["index_code"].eq(code)].copy()
    factor_frame, state_lags = screen.build_factor_frame(target, contract, profiles, STATE_QUALITY)
    factor_frame, registry = make_factor_catalog(factor_frame, contract, profiles)
    fast = fields(registry, "fast")
    slow = fields(registry, "slow")
    matrix = pd.concat(
        [
            factor_frame[fast].rename(columns=lambda c: f"fast::{c}"),
            factor_frame[slow].rename(columns=lambda c: f"slow::{c}"),
        ],
        axis=1,
    )
    group = target.sort_values("trade_date").set_index("trade_date")
    first_valid = group.index[group["close"].notna()].min()
    max_horizon = max(cfg["fast_horizons"] + cfg["slow_horizons"])
    development_start = group.index[group.index.get_loc(first_valid) + cfg["train_days"] + max_horizon]
    development_end = pd.Timestamp(cfg["oos_start"]) - pd.Timedelta(days=1)
    replay_start = pd.Timestamp(cfg["oos_start"])
    replay_end = group.index.max()
    rows, ledgers, folds, selections = [], [], [], []
    for fast_horizon in cfg["fast_horizons"]:
        for slow_horizon in cfg["slow_horizons"]:
            for fold_days in cfg["fold_days"]:
                for scheme in SCHEMES:
                    for direction in cfg["directions"]:
                        train, train_folds, train_selections = run_fold(
                            group, matrix, scheme, direction, fast_horizon, slow_horizon, fold_days,
                            cfg["train_days"], cfg["top_k"], development_start, development_end, "development",
                        )
                        replay, replay_folds, replay_selections = run_fold(
                            group, matrix, scheme, direction, fast_horizon, slow_horizon, fold_days,
                            cfg["train_days"], cfg["top_k"], replay_start, replay_end, "historical_replay",
                        )
                        tm = worker.metrics(train) if {"strategy_return", "benchmark_return"}.issubset(train.columns) else empty_metrics()
                        rm = worker.metrics(replay) if {"strategy_return", "benchmark_return"}.issubset(replay.columns) else empty_metrics()
                        row = {
                            "index_code": code,
                            "index_name": profile["name"],
                            "scheme": scheme,
                            "scheme_label": SCHEMES[scheme],
                            "direction": direction,
                            "fast_horizon_days": fast_horizon,
                            "slow_horizon_days": slow_horizon,
                            "fold_days": fold_days,
                            "development_rows": tm["rows"],
                            "replay_rows": rm["rows"],
                        }
                        for prefix, metrics in (("development", tm), ("replay", rm)):
                            for key in (
                                "annualized_return", "benchmark_annualized_return", "annualized_excess",
                                "active_sharpe", "max_drawdown", "annual_turnover", "total_cost",
                            ):
                                row[f"{prefix}_{key}"] = metrics.get(key)
                        row["development_pass"] = bool(tm["annualized_excess"] > 0 and tm["active_sharpe"] > 0)
                        row["replay_pass"] = bool(rm["annualized_excess"] > 0 and rm["active_sharpe"] > 0)
                        row["development_score"] = float(min(tm["annualized_excess"], tm["active_sharpe"]))
                        row["replay_score"] = float(min(rm["annualized_excess"], rm["active_sharpe"]))
                        rows.append(row)
                        if not train.empty:
                            ledgers.append(train.assign(period="development", scheme=scheme, direction=direction, fast_horizon_days=fast_horizon, slow_horizon_days=slow_horizon, fold_days=fold_days).reset_index())
                        if not replay.empty:
                            ledgers.append(replay.assign(period="historical_replay", scheme=scheme, direction=direction, fast_horizon_days=fast_horizon, slow_horizon_days=slow_horizon, fold_days=fold_days).reset_index())
                        folds.extend(train_folds + replay_folds)
                        selections.extend(train_selections + replay_selections)

    out = output_root / code
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "slow_adjustment_results.csv", index=False, encoding="utf-8-sig")
    if ledgers:
        pd.concat(ledgers, ignore_index=True).to_csv(out / "slow_adjustment_ledgers.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folds).to_csv(out / "fold_audit.csv", index=False, encoding="utf-8-sig")
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(out / "selected_features_by_fold.csv", index=False, encoding="utf-8-sig")
    registry.to_csv(out / "factor_catalog.csv", index=False, encoding="utf-8-sig")
    if not selection_frame.empty:
        contribution = (
            selection_frame.groupby(["scheme", "direction", "role", "field"]).size()
            .rename("selected_fold_count").reset_index()
            .merge(registry[["field", "source", "source_label", "family", "model_role", "transform", "base_field"]],
                   on="field", how="left")
            .sort_values(["scheme", "direction", "role", "selected_fold_count"], ascending=[True, True, True, False])
        )
    else:
        contribution = pd.DataFrame()
    contribution.to_csv(out / "factor_contribution.csv", index=False, encoding="utf-8-sig")
    ranked = result.sort_values(["development_pass", "development_score", "replay_score"], ascending=[False, False, False])
    summary = {
        "index": code,
        "index_name": profile["name"],
        "development_interval": [development_start.date().isoformat(), development_end.date().isoformat()],
        "historical_replay_interval": [replay_start.date().isoformat(), replay_end.date().isoformat()],
        "maturity": "historical_replay_not_new_oos",
        "fast_feature_count": len(fast),
        "slow_feature_count": len(slow),
        "fast_horizons": cfg["fast_horizons"],
        "slow_horizons": cfg["slow_horizons"],
        "fold_days": cfg["fold_days"],
        "best_by_development": ranked.iloc[0].to_dict() if not ranked.empty else None,
        "best_replay_positive": (
            result.loc[result["replay_pass"]].sort_values("replay_score", ascending=False).iloc[0].to_dict()
            if result["replay_pass"].any() else None
        ),
        "schemes": SCHEMES,
        "state_lags": state_lags,
        "cost": {"commission_one_way": COMMISSION, "slippage_one_way": SLIPPAGE, "total_one_way": COMMISSION + SLIPPAGE},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=list(SLOW_CONFIG))
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    contract = pd.read_csv(CONTRACT)
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    codes = [args.index] if args.index else list(SLOW_CONFIG)
    summaries = [run_index(code, panel, contract, profiles, args.output_root) for code in codes]
    print(json.dumps({"output_root": str(args.output_root), "indices": summaries}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

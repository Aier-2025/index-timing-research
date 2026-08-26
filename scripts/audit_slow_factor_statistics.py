"""Audit slow-factor statistical predictability before model construction.

This script is descriptive. It does not select a trading direction, tune a
position rule, or claim new out-of-sample evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import screen_index_timing_factors as screen


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
CONTRACT = Path(r"F:\量化因子库\docs\reports\csi1000_factor_term_audit_20260819\feature_contract_draft.csv")
STATE_QUALITY = Path(r"F:\量化因子库\docs\factor_library\market_state_factor_quality.csv")
PROFILES = ROOT / "config" / "index_research_profiles.json"
OUT = DATA_DIR / "slow_factor_statistics_20260826"

INDICES = ["000905", "000852", "399006", "000688"]
HORIZONS = [5, 10, 20, 60]
WINDOWS = [252, 504]
SLOW_BUCKETS = {"persistent_daily", "lagged_persistent", "slow_release"}


def spearman(x: pd.Series, y: pd.Series, min_obs: int = 60) -> float:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_obs or pair.iloc[:, 0].nunique() < 8 or pair.iloc[:, 1].nunique() < 8:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def future_return(group: pd.DataFrame, horizon: int) -> pd.Series:
    one_day = pd.to_numeric(group["next_open_to_open"], errors="coerce")
    return (1.0 + one_day).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1))


def qspread(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 60 or pair.iloc[:, 0].nunique() < 8:
        return np.nan, np.nan, np.nan, np.nan
    ranks = pair.iloc[:, 0].rank(pct=True)
    bottom = pair.loc[ranks <= 0.2, pair.columns[1]].mean()
    top = pair.loc[ranks >= 0.8, pair.columns[1]].mean()
    return float(bottom), float(top), float(top - bottom), float(pair.iloc[:, 1].mean())


def annual_stats(x: pd.Series, y: pd.Series, horizon: int) -> dict[str, float]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if frame.empty:
        return {"years": 0, "ic_mean": np.nan, "ic_std": np.nan, "positive_ratio": np.nan, "sign_consistency": np.nan}
    values = []
    for _, block in frame.groupby(frame.index.year):
        value = spearman(block["x"], block["y"], min_obs=max(40, horizon * 4))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"years": 0, "ic_mean": np.nan, "ic_std": np.nan, "positive_ratio": np.nan, "sign_consistency": np.nan}
    values = np.asarray(values, dtype=float)
    positive = float((values > 0).mean())
    negative = float((values < 0).mean())
    return {
        "years": int(len(values)),
        "ic_mean": float(values.mean()),
        "ic_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "positive_ratio": positive,
        "sign_consistency": max(positive, negative),
    }


def rolling_stats(x: pd.Series, y: pd.Series, window: int) -> dict[str, float]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < window + 20:
        return {"windows": 0, "ic_mean": np.nan, "ic_std": np.nan, "positive_ratio": np.nan}
    values = []
    for end in range(window, len(frame) + 1, 21):
        value = spearman(frame["x"].iloc[end - window:end], frame["y"].iloc[end - window:end], min_obs=60)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"windows": 0, "ic_mean": np.nan, "ic_std": np.nan, "positive_ratio": np.nan}
    values = np.asarray(values, dtype=float)
    return {
        "windows": int(len(values)),
        "ic_mean": float(values.mean()),
        "ic_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "positive_ratio": float((values > 0).mean()),
    }


def role_label(ic20: float, ic60: float, q20: float, q60: float) -> str:
    long_signal = np.nanmean([ic20, ic60])
    spread = np.nanmean([q20, q60])
    if np.isfinite(long_signal) and np.isfinite(spread):
        if abs(long_signal) >= 0.03 and abs(spread) >= 0.01:
            return "方向候选"
        if abs(long_signal) < 0.03 and abs(spread) >= 0.01:
            return "状态/风险候选"
    return "统计弱或不稳定"


def run_index(code: str, panel: pd.DataFrame, contract: pd.DataFrame, profiles: dict, output_root: Path) -> dict:
    target = panel.loc[panel["index_code"].eq(code)].copy()
    group = target.sort_values("trade_date").set_index("trade_date")
    factor_frame, state_lags = screen.build_factor_frame(target, contract, profiles, STATE_QUALITY)
    contract_map = contract.set_index("field")
    rows = []
    slow_fields = []
    for field in factor_frame.columns:
        if field == "trade_date" or field not in contract_map.index:
            continue
        item = contract_map.loc[field]
        if str(item.horizon_bucket) not in SLOW_BUCKETS:
            continue
        slow_fields.append(field)

    target_map = {h: future_return(group, h) for h in HORIZONS}
    for field in slow_fields:
        series = pd.to_numeric(factor_frame.set_index("trade_date")[field], errors="coerce").reindex(group.index)
        valid = series.dropna()
        row = {
            "index_code": code,
            "index_name": profiles[code]["name"],
            "field": field,
            "source": str(contract_map.loc[field, "source"]),
            "family": str(contract_map.loc[field, "family"]),
            "horizon_bucket": str(contract_map.loc[field, "horizon_bucket"]),
            "observed_frequency": str(contract_map.loc[field, "observed_frequency"]),
            "availability_lag_days": int(contract_map.loc[field, "availability_lag_days"]),
            "n_obs": int(valid.size),
            "coverage": float(series.notna().mean()),
            "unique_values": int(valid.nunique()),
            "autocorr_1d": float(series.autocorr(1)) if len(valid) > 3 else np.nan,
            "autocorr_20d": float(series.autocorr(20)) if len(valid) > 25 else np.nan,
            "update_rate": float(series.diff().abs().gt(1e-12).mean()),
        }
        q = {}
        for horizon, target_y in target_map.items():
            row[f"rank_ic_{horizon}d"] = spearman(series, target_y, min_obs=max(60, horizon * 4))
            bottom, top, spread, target_mean = qspread(series, target_y)
            row[f"q1_{horizon}d"] = bottom
            row[f"q5_{horizon}d"] = top
            row[f"q5_minus_q1_{horizon}d"] = spread
            row[f"target_mean_{horizon}d"] = target_mean
            annual = annual_stats(series, target_y, horizon)
            for key, value in annual.items():
                row[f"annual_{horizon}d_{key}"] = value
            for window in WINDOWS:
                rolling = rolling_stats(series, target_y, window)
                for key, value in rolling.items():
                    row[f"rolling_{window}_{horizon}d_{key}"] = value
            q[horizon] = row[f"q5_minus_q1_{horizon}d"]
        row["statistical_role"] = role_label(
            row["rank_ic_20d"], row["rank_ic_60d"], q[20], q[60]
        )
        rows.append(row)

    profile = pd.DataFrame(rows)
    if profile.empty:
        return {"index": code, "index_name": profiles[code]["name"], "slow_fields": 0}
    out = output_root / code
    out.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out / "slow_field_statistics.csv", index=False, encoding="utf-8-sig")

    summary = profile.groupby(["source", "family", "horizon_bucket"], dropna=False).agg(
        fields=("field", "size"),
        coverage_median=("coverage", "median"),
        update_rate_median=("update_rate", "median"),
        autocorr_1d_median=("autocorr_1d", "median"),
        autocorr_20d_median=("autocorr_20d", "median"),
        ic20_abs_median=("rank_ic_20d", lambda x: x.abs().median()),
        ic60_abs_median=("rank_ic_60d", lambda x: x.abs().median()),
        annual20_sign_consistency=("annual_20d_sign_consistency", "median"),
        annual60_sign_consistency=("annual_60d_sign_consistency", "median"),
        direction_candidates=("statistical_role", lambda x: int((x == "方向候选").sum())),
        state_candidates=("statistical_role", lambda x: int((x == "状态/风险候选").sum())),
    ).reset_index()
    summary.to_csv(out / "slow_family_statistics.csv", index=False, encoding="utf-8-sig")

    corr = pd.DataFrame({field: pd.to_numeric(factor_frame.set_index("trade_date")[field], errors="coerce") for field in slow_fields})
    corr_level = corr.corr(min_periods=120).abs()
    corr_change = corr.diff(5).corr(min_periods=120).abs()
    redundancy = []
    for name, matrix in (("level", corr_level), ("five_day_change", corr_change)):
        values = matrix.to_numpy()
        values = values[np.triu_indices_from(values, k=1)]
        values = values[np.isfinite(values)]
        redundancy.append({
            "basis": name,
            "fields": len(slow_fields),
            "pair_count": int(len(values)),
            "median_abs_correlation": float(np.median(values)) if len(values) else np.nan,
            "p90_abs_correlation": float(np.quantile(values, 0.9)) if len(values) else np.nan,
            "share_abs_corr_over_080": float((values > 0.8).mean()) if len(values) else np.nan,
            "share_abs_corr_over_095": float((values > 0.95).mean()) if len(values) else np.nan,
        })
    pd.DataFrame(redundancy).to_csv(out / "slow_redundancy.csv", index=False, encoding="utf-8-sig")

    result = {
        "index": code,
        "index_name": profiles[code]["name"],
        "slow_fields": len(slow_fields),
        "field_rows": len(profile),
        "role_counts": profile["statistical_role"].value_counts().to_dict(),
        "source_summary": summary.to_dict("records"),
        "redundancy": redundancy,
        "state_lags": state_lags,
        "horizons": HORIZONS,
        "rolling_windows": WINDOWS,
        "interpretation": "descriptive_statistics_only_no_model_selection_no_new_oos_claim",
    }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=INDICES)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str}, low_memory=False)
    contract = pd.read_csv(CONTRACT)
    codes = [args.index] if args.index else INDICES
    results = [run_index(code, panel, contract, profiles, args.output_root) for code in codes]
    print(json.dumps({"output_root": str(args.output_root), "indices": results}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

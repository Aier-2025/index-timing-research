"""Screen index-specific timing candidates under strict timestamp contracts.

This is a descriptive candidate screen, not a model-selection or OOS claim.
Each index has its own allowed families, horizons, and candidate budget.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
MARKET_STATE = Path(r"F:\量化因子库\docs\factor_library\market_state_factor_panel.csv")
CONTRACT = Path(r"F:\量化因子库\docs\reports\csi1000_factor_term_audit_20260819\feature_contract_draft.csv")
PROFILES = ROOT / "config" / "index_research_profiles.json"
OUT = DATA_DIR / "index_factor_screen"


def parse_horizons(value):
    return {int(x) for x in str(value).split(",") if str(x).strip().isdigit()}


def spearman(x, y):
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    return x[valid].rank().corr(y[valid].rank())


def future_return(group, horizon):
    one_day = pd.to_numeric(group["next_open_to_open"], errors="coerce")
    return (1.0 + one_day).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1.0


def is_suspect(name, patterns):
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def add_market_state(factors, state_quality):
    if not MARKET_STATE.exists():
        return factors, {}
    state = pd.read_csv(MARKET_STATE, parse_dates=["date"])
    state = state.rename(columns={"date": "trade_date"}).set_index("trade_date").sort_index()
    factor_frame = factors.set_index("trade_date")
    quality = pd.read_csv(state_quality) if state_quality.exists() else pd.DataFrame()
    lag_map = dict(zip(quality.get("factor_id", []), quality.get("lag_trading_days", [])))
    for column in state.columns:
        lag = int(lag_map.get(column, 0) or 0)
        factor_frame[column] = pd.to_numeric(state[column], errors="coerce").shift(lag)
    return factor_frame.reset_index(), {column: int(lag_map.get(column, 0) or 0) for column in state.columns}


def build_factor_frame(panel, contract, profiles, state_quality):
    date_index = pd.DataFrame({"trade_date": sorted(panel["trade_date"].dropna().unique())})
    values = {"trade_date": date_index["trade_date"].to_numpy()}
    contract_fields = set(contract["field"].astype(str))
    technical = [column for column in panel.columns if column.startswith(("mom_", "close_to_ma_", "vol_", "drawdown_", "intraday_", "close_location"))]
    unique_panel = panel.drop_duplicates("trade_date").set_index("trade_date")
    for column in technical:
        values[column] = unique_panel[column].reindex(date_index["trade_date"].values).to_numpy()

    for row in contract.itertuples(index=False):
        field = str(row.field)
        if field not in unique_panel.columns:
            continue
        series_values = pd.to_numeric(unique_panel[field], errors="coerce").reindex(date_index["trade_date"].values)
        lag = int(row.availability_lag_days) if pd.notna(row.availability_lag_days) else 0
        series = pd.Series(series_values.to_numpy(), index=date_index.index).shift(lag)
        if field.startswith(("stock_macro_", "public_macro_")):
            series.loc[pd.to_datetime(date_index["trade_date"]) < pd.Timestamp("2015-01-01")] = np.nan
        values[field] = series.to_numpy()
    factor_frame = pd.DataFrame(values)
    factor_frame, state_lags = add_market_state(factor_frame, state_quality)
    return factor_frame, state_lags


def candidate_registry(contract, factor_frame, profile, common):
    patterns = common["exclude_name_patterns"]
    allowed = set(profile["allowed_contract_families"])
    contract_map = contract.set_index("field")
    rows = []
    for field in factor_frame.columns:
        if field == "trade_date" or is_suspect(field, patterns):
            continue
        if field.startswith(("mom_", "close_to_ma_", "vol_", "drawdown_", "intraday_", "close_location")):
            rows.append({"field": field, "family": "price_technical", "availability_lag_days": 0, "source": "project_ohlc", "allowed_horizons": set(common["horizons_days"])})
            continue
        if field.startswith(tuple(profile["allowed_market_state_prefixes"])):
            rows.append({"field": field, "family": "market_state", "availability_lag_days": 0, "source": "market_state_factor_panel", "allowed_horizons": set(common["horizons_days"])})
            continue
        if field not in contract_map.index:
            continue
        item = contract_map.loc[field]
        if str(item.family) not in allowed:
            continue
        rows.append({"field": field, "family": str(item.family), "availability_lag_days": int(item.availability_lag_days), "source": str(item.source), "allowed_horizons": parse_horizons(item.allowed_research_horizons_days)})
    return rows


def screen_one(index_code, prices, factor_frame, registry, profile, common):
    group = prices.loc[prices["index_code"].eq(index_code)].sort_values("trade_date").copy().set_index("trade_date")
    if group.empty:
        return []
    result = []
    for horizon in profile["horizons_days"]:
        label = future_return(group.reset_index(), horizon)
        label.index = group.index
        allowed_items = [item for item in registry if horizon in item["allowed_horizons"]]
        if not allowed_items:
            continue
        fields = [item["field"] for item in allowed_items]
        factor_table = factor_frame.set_index("trade_date")[fields].reindex(group.index)
        label = label.replace([np.inf, -np.inf], np.nan)
        factor_table = factor_table.replace([np.inf, -np.inf], np.nan).where(label.notna(), np.nan)
        factor_table = factor_table.loc[:, factor_table.nunique(dropna=True).gt(1)]
        if factor_table.empty:
            continue
        n_obs = factor_table.notna().sum()
        factor_table = factor_table.loc[:, n_obs.ge(common["min_total_observations"])]
        if factor_table.empty:
            continue
        n_obs = factor_table.notna().sum()
        coverage = n_obs / len(label)
        rank_table = factor_table.rank(axis=0, method="average")
        daily_rank_ic = rank_table.corrwith(label.rank())

        annual_rows = {}
        for year in sorted(set(factor_table.index.year)):
            mask = factor_table.index.year == year
            year_factors = factor_table.loc[mask]
            year_label = label.loc[mask]
            year_counts = year_factors.notna().sum()
            year_ic = pd.Series(np.nan, index=year_factors.columns, dtype=float)
            if year_label.nunique(dropna=True) > 1:
                variable = year_factors.nunique(dropna=True).gt(1)
                if variable.any():
                    year_ic.loc[variable] = year_factors.loc[:, variable].rank(axis=0, method="average").corrwith(year_label.rank())
            year_ic.loc[year_counts.lt(common["annual_min_observations"])] = np.nan
            annual_rows[year] = year_ic
        annual = pd.DataFrame(annual_rows).T
        year_count = annual.notna().sum()
        year_mean = annual.mean()
        year_std = annual.std(ddof=1)
        year_positive = annual.gt(0).sum().div(year_count.replace(0, np.nan))
        year_negative = annual.lt(0).sum().div(year_count.replace(0, np.nan))

        pct_rank = factor_table.rank(pct=True, method="average")
        upper_mask = pct_rank.ge(0.8)
        lower_mask = pct_rank.le(0.2)
        upper_mean = upper_mask.mul(label, axis=0).sum().div(upper_mask.sum().replace(0, np.nan))
        lower_mean = lower_mask.mul(label, axis=0).sum().div(lower_mask.sum().replace(0, np.nan))
        qspread = upper_mean - lower_mean
        meta = {item["field"]: item for item in allowed_items}
        for field in factor_table.columns:
            item = meta[field]
            rank_ic = daily_rank_ic[field]
            result.append({
                "index_code": index_code,
                "index_name": profile["name"],
                "horizon_days": horizon,
                "field": field,
                "family": item["family"],
                "source": item["source"],
                "n_obs": int(n_obs[field]),
                "coverage": float(coverage[field]),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "year_ic_mean": float(year_mean[field]) if pd.notna(year_mean[field]) else np.nan,
                "year_ic_std": float(year_std[field]) if pd.notna(year_std[field]) else np.nan,
                "year_icir": float(year_mean[field] / year_std[field]) if pd.notna(year_mean[field]) and pd.notna(year_std[field]) and year_std[field] else np.nan,
                "year_count": int(year_count[field]),
                "year_positive_ratio": float(year_positive[field]) if pd.notna(year_positive[field]) else np.nan,
                "direction_stability": float(max(year_positive[field], year_negative[field])) if pd.notna(year_positive[field]) and pd.notna(year_negative[field]) else np.nan,
                "q5_minus_q1": float(qspread[field]) if pd.notna(qspread[field]) else np.nan,
                "direction": "direct" if pd.notna(rank_ic) and rank_ic >= 0 else "inverse",
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=["000905", "000852", "399006", "000688", "932000"])
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    common = profiles["common"]
    selected_profiles = {args.index: profiles[args.index]} if args.index else {key: value for key, value in profiles.items() if key != "common"}
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    contract = pd.read_csv(CONTRACT)
    all_rows = []
    top_rows = []
    state_lags = {}
    for code, profile in selected_profiles.items():
        target_panel = panel.loc[panel["index_code"].eq(code)].copy()
        factor_frame, state_lags = build_factor_frame(target_panel, contract, profiles, Path(r"F:\量化因子库\docs\factor_library\market_state_factor_quality.csv"))
        registry = candidate_registry(contract, factor_frame, profile, common)
        rows = screen_one(code, target_panel, factor_frame, registry, profile, common)
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame["eligible_for_candidate"] = (frame["n_obs"] >= common["min_total_observations"]) & (frame["direction_stability"] >= common["min_year_positive_ratio"])
        frame["screen_score"] = frame["year_ic_mean"].abs() * frame["direction_stability"] * np.log1p(frame["n_obs"])
        selected = frame.loc[frame["eligible_for_candidate"]].sort_values(["horizon_days", "screen_score"], ascending=[True, False]).groupby("horizon_days").head(int(profile["max_candidates_per_horizon"]))
        all_rows.extend(frame.to_dict("records"))
        top_rows.extend(selected.to_dict("records"))
    output_dir = args.output_root / args.index if args.index else args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frame = pd.DataFrame(all_rows)
    top_frame = pd.DataFrame(top_rows)
    all_frame.to_csv(output_dir / "all_factor_screen.csv", index=False, encoding="utf-8-sig")
    top_frame.to_csv(output_dir / "index_specific_candidates.csv", index=False, encoding="utf-8-sig")
    manifest = {"index": args.index or "all", "profiles": str(PROFILES), "contract_rows": int(len(contract)), "factor_columns_screened": int(len(set(all_frame["field"])) if not all_frame.empty else 0), "candidate_rows": int(len(top_frame)), "state_lags": state_lags, "rules": {"macro_start": common["macro_sentiment_start"], "label": "T+1 open to T+2 open compounded by horizon", "screen_only": True}}
    (output_dir / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Validate date alignment, factor admission, and no-lookahead panel rules."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
OUT = DATA_DIR / "index_timing_panel_validation.json"
INDEX_CODES = ["000905", "399006", "000852", "000688", "932000"]


def main():
    frame = pd.read_csv(PANEL, parse_dates=["trade_date", "execution_date", "signal_date"], dtype={"index_code": str})
    checks = {}
    checks["row_count"] = int(len(frame))
    checks["duplicate_index_dates"] = int(frame.duplicated(["index_code", "trade_date"]).sum())
    checks["index_codes"] = sorted(frame["index_code"].dropna().unique().tolist())
    checks["expected_codes_present"] = set(INDEX_CODES).issubset(set(checks["index_codes"]))
    checks["union_date_min"] = frame["trade_date"].min().date().isoformat()
    checks["union_date_max"] = frame["trade_date"].max().date().isoformat()

    pre_macro = frame.loc[frame["trade_date"] < pd.Timestamp("2015-01-01")]
    factor_columns = [c for c in frame.columns if "stock_macro_" in c or c.startswith("public_macro_") or c.startswith("flow.") or c.startswith("breadth.")]
    checks["pre_2015_macro_nonnull_cells"] = int(pre_macro[factor_columns].notna().sum().sum()) if factor_columns else 0
    checks["pre_2015_macro_factor_count"] = int(len(factor_columns))
    checks["pre_2015_macro_clean"] = checks["pre_2015_macro_nonnull_cells"] == 0

    # Validate that the next-open target is aligned to the two following opens.
    ordered = frame.sort_values(["index_code", "trade_date"])
    expected = ordered.groupby("index_code")["open"].shift(-2).div(ordered.groupby("index_code")["open"].shift(-1)).sub(1)
    valid = ordered["next_open_to_open"].notna() & expected.notna()
    checks["next_open_alignment_max_abs_error"] = float((ordered.loc[valid, "next_open_to_open"] - expected.loc[valid]).abs().max()) if valid.any() else None
    checks["next_open_alignment_clean"] = checks["next_open_alignment_max_abs_error"] is not None and checks["next_open_alignment_max_abs_error"] < 1e-12

    # A 20-day momentum value at row t may only use the current and prior 20 closes.
    sample = ordered.loc[ordered["mom_20d"].notna()].groupby("index_code").head(1)
    momentum_errors = []
    for row in sample.itertuples():
        history = ordered.loc[(ordered["index_code"] == row.index_code) & (ordered["trade_date"] <= row.trade_date), "close"].dropna()
        if len(history) >= 21:
            momentum_errors.append(abs(float(row.mom_20d) - float(history.iloc[-1] / history.iloc[-21] - 1)))
    checks["momentum_alignment_max_abs_error"] = float(max(momentum_errors)) if momentum_errors else None
    checks["momentum_alignment_clean"] = checks["momentum_alignment_max_abs_error"] is not None and max(momentum_errors) < 1e-12

    checks["passed"] = all([
        checks["duplicate_index_dates"] == 0,
        checks["expected_codes_present"],
        checks["pre_2015_macro_clean"],
        checks["next_open_alignment_clean"],
        checks["momentum_alignment_clean"],
    ])
    OUT.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

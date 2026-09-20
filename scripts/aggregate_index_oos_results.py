"""Aggregate isolated index OOS worker summaries without selecting a winner."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"F:\data\index_timing_raw\index_oos_workers")
OUT = ROOT / "aggregate"


def main():
    rows = []
    worker_summaries = []
    for index_dir in sorted(ROOT.iterdir()):
        if not index_dir.is_dir() or index_dir.name == "aggregate":
            continue
        path = index_dir / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        worker_summaries.append(summary)
        for horizon, periods in summary.get("horizons", {}).items():
            row = {"index_code": summary["index"], "index_name": summary["index_name"], "horizon_days": int(horizon)}
            for period in ("training", "backtest"):
                values = periods.get(period, {}).get("metrics", {})
                interval = periods.get(period, {}).get("interval", {})
                row[f"{period}_rows"] = values.get("rows", 0)
                row[f"{period}_signal_start"] = interval.get("signal_start")
                row[f"{period}_signal_end"] = interval.get("signal_end")
                row[f"{period}_annualized_excess"] = values.get("annualized_excess")
                row[f"{period}_active_sharpe"] = values.get("active_sharpe")
                row[f"{period}_annualized_return"] = values.get("annualized_return")
                row[f"{period}_benchmark_annualized_return"] = values.get("benchmark_annualized_return")
                row[f"{period}_annual_turnover"] = values.get("annual_turnover")
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows).sort_values(["index_code", "horizon_days"])
    metrics.to_csv(OUT / "oos_metrics_all.csv", index=False, encoding="utf-8-sig")
    if not metrics.empty:
        metrics["oos_candidate_flag"] = (metrics["backtest_annualized_excess"] > 0) & (metrics["backtest_active_sharpe"] > 0)
        metrics.to_csv(OUT / "oos_metrics_all.csv", index=False, encoding="utf-8-sig")
    report = {
        "worker_count": len(worker_summaries),
        "horizon_rows": int(len(metrics)),
        "positive_excess_and_active_sharpe_rows": int(metrics["oos_candidate_flag"].sum()) if not metrics.empty else 0,
        "workers": worker_summaries,
        "interpretation": "Independent rolling OOS diagnostics; no horizon or model was selected from the OOS results.",
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"worker_count": len(worker_summaries), "horizon_rows": int(len(metrics)), "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

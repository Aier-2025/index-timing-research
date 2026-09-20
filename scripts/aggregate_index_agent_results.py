"""Aggregate isolated per-index screening worker outputs."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"F:\data\index_timing_raw\index_factor_screen")
OUT = ROOT / "aggregate"


def main():
    all_frames = []
    summaries = []
    for index_dir in sorted(ROOT.iterdir()):
        if not index_dir.is_dir() or index_dir.name == "aggregate":
            continue
        candidate = index_dir / "index_specific_candidates.csv"
        summary = index_dir / "summary.json"
        if candidate.exists():
            all_frames.append(pd.read_csv(candidate, dtype={"index_code": str}))
        if summary.exists():
            summaries.append(json.loads(summary.read_text(encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    candidates.to_csv(OUT / "index_specific_candidates_all.csv", index=False, encoding="utf-8-sig")
    family = candidates.groupby(["index_code", "horizon_days", "family"], as_index=False).size() if not candidates.empty else pd.DataFrame()
    family.to_csv(OUT / "candidate_family_counts.csv", index=False, encoding="utf-8-sig")
    horizon = candidates.groupby(["index_code", "horizon_days"], as_index=False).agg(candidate_rows=("field", "size"), median_abs_rank_ic=("rank_ic", lambda x: x.abs().median())) if not candidates.empty else pd.DataFrame()
    horizon.to_csv(OUT / "candidate_horizon_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "worker_count": len(summaries),
        "candidate_rows": int(len(candidates)),
        "workers": summaries,
        "interpretation": "Descriptive full-sample screening only; candidate selection must be repeated inside rolling development folds.",
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"worker_count": len(summaries), "candidate_rows": int(len(candidates)), "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

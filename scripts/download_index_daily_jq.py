"""Download raw daily price-index data from JoinQuant.

This script is executed in a JoinQuant research notebook by the local runner.
It intentionally keeps the price-index level unadjusted and records failures
per index so a missing code cannot be mistaken for an empty successful file.
"""

import json
import os
import zipfile
from datetime import date

import pandas as pd


START_DATE = date(2004, 12, 31)
END_DATE = date.today()
ARCHIVE_NAME = "index_timing_daily_raw_jq_20260824.zip"
FIELDS = ["open", "high", "low", "close", "volume", "money"]

INDICES = {
    "000852": {
        "name_zh": "中证1000",
        "candidates": ["000852.XSHG", "000852.SH"],
        "expected_start": "2005-01-04",
    },
    "000905": {
        "name_zh": "中证500",
        "candidates": ["000905.XSHG", "000905.SH"],
        "expected_start": "2005-01-04",
    },
    "399006": {
        "name_zh": "创业板指",
        "candidates": ["399006.XSHE", "399006.SZ"],
        "expected_start": "2010-06-01",
    },
    "000688": {
        "name_zh": "科创50",
        "candidates": ["000688.XSHG", "000688.SH"],
        "expected_start": "2020-07-23",
    },
    "932000": {
        "name_zh": "中证2000",
        "candidates": ["932000.CSI", "932000.XSHG", "932000.SH"],
        "expected_start": "2023-08-11",
    },
}


def normalize(frame, code, jq_code):
    columns = ["trade_date", "code", "jq_code", *FIELDS]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.reset_index().copy()
    time_column = next(
        (column for column in result.columns if column in {"time", "date", "datetime", "index"}),
        result.columns[0],
    )
    result["trade_date"] = pd.to_datetime(result[time_column], errors="coerce").dt.strftime("%Y-%m-%d")
    result["code"] = code
    result["jq_code"] = jq_code
    for field in FIELDS:
        if field not in result:
            result[field] = pd.NA
    result = result[["trade_date", "code", "jq_code", *FIELDS]]
    result = result.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    return result.sort_values("trade_date").reset_index(drop=True)


def query_index(candidates):
    issues = []
    for jq_code in candidates:
        try:
            frame = get_price(
                jq_code,
                start_date=START_DATE.isoformat(),
                end_date=END_DATE.isoformat(),
                frequency="daily",
                fields=FIELDS,
                skip_paused=False,
                panel=False,
                fq=None,
            )
            normalized = normalize(frame, jq_code.split(".")[0], jq_code)
            if normalized.empty:
                issues.append({"jq_code": jq_code, "detail": "empty_result"})
                continue
            return normalized, jq_code, issues
        except Exception as exc:
            issues.append({"jq_code": jq_code, "detail": repr(exc)})
    return pd.DataFrame(columns=["trade_date", "code", "jq_code", *FIELDS]), "", issues


def quality(frame, spec, jq_code, issues):
    numeric = frame[FIELDS] if not frame.empty else pd.DataFrame(columns=FIELDS)
    close = pd.to_numeric(numeric["close"], errors="coerce") if not frame.empty else pd.Series(dtype=float)
    return {
        "status": "ok" if not frame.empty and close.notna().any() else "failed",
        "jq_code": jq_code,
        "name_zh": spec["name_zh"],
        "expected_start": spec["expected_start"],
        "rows": int(len(frame)),
        "first_date": str(frame["trade_date"].iloc[0]) if not frame.empty else "",
        "last_date": str(frame["trade_date"].iloc[-1]) if not frame.empty else "",
        "duplicate_dates": int(frame.duplicated("trade_date").sum()),
        "null_counts": {field: int(numeric[field].isna().sum()) for field in FIELDS},
        "close_valid_rows": int(close.notna().sum()),
        "issues": issues,
    }


def main() -> None:
    manifest = {
        "source": "JoinQuant get_price",
        "retrieved_on": END_DATE.isoformat(),
        "adjustment": "raw_index_level_fq_none",
        "start_requested": START_DATE.isoformat(),
        "end_requested": END_DATE.isoformat(),
        "fields": FIELDS,
        "indices": {},
    }
    archive_tmp = ARCHIVE_NAME + ".part"
    with zipfile.ZipFile(archive_tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for code, spec in INDICES.items():
            frame, jq_code, issues = query_index(list(spec["candidates"]))
            payload = frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            archive.writestr(f"daily/{code}.csv", payload)
            manifest["indices"][code] = quality(frame, spec, jq_code, issues)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    os.replace(archive_tmp, ARCHIVE_NAME)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

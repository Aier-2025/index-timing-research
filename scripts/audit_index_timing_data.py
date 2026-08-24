"""Audit and normalize the five-index daily raw data kept under F:/data."""

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


DATA_DIR = Path(r"F:\data\index_timing_raw")
JQ_ARCHIVE = DATA_DIR / "index_timing_daily_raw_jq_20260824.zip"
TONG_LIAN_FILE = DATA_DIR / "932000_中证2000_daily_tonglian.csv"
OUT_PANEL = DATA_DIR / "index_timing_daily_panel.csv"
OUT_MANIFEST = DATA_DIR / "index_timing_data_audit.json"

INDEX_META = {
    "000905": {"name": "中证500", "source": "joinquant"},
    "399006": {"name": "创业板指", "source": "joinquant"},
    "000852": {"name": "中证1000", "source": "joinquant"},
    "000688": {"name": "科创50", "source": "joinquant"},
    "932000": {"name": "中证2000", "source": "tonglian"},
}
PRICE_FIELDS = ["open", "high", "low", "close"]
ALL_FIELDS = [*PRICE_FIELDS, "volume", "money"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_joinquant():
    if not JQ_ARCHIVE.exists():
        raise FileNotFoundError(JQ_ARCHIVE)
    frames = {}
    with zipfile.ZipFile(JQ_ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("JoinQuant archive failed CRC validation")
        for code in ["000905", "399006", "000852", "000688", "932000"]:
            name = f"daily/{code}.csv"
            if name in archive.namelist():
                frames[code] = pd.read_csv(archive.open(name), dtype={"code": str})
    return frames


def normalize(frame, code, source):
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result["index_code"] = code
    result["source"] = source
    for field in ALL_FIELDS:
        if field not in result:
            result[field] = pd.NA
        result[field] = pd.to_numeric(result[field], errors="coerce")
    result = result[["trade_date", "index_code", "source", *ALL_FIELDS]]
    return result.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates(["trade_date", "index_code"], keep="last")


def audit_frame(frame):
    frame = frame.sort_values("trade_date")
    valid = frame.loc[frame["close"].notna()]
    gaps = frame.loc[frame["close"].notna(), "trade_date"].diff().dt.days.dropna()
    returns = valid["close"].pct_change()
    return {
        "rows": int(len(frame)),
        "first_row_date": frame["trade_date"].min().date().isoformat() if len(frame) else "",
        "last_row_date": frame["trade_date"].max().date().isoformat() if len(frame) else "",
        "first_valid_close_date": valid["trade_date"].min().date().isoformat() if len(valid) else "",
        "last_valid_close_date": valid["trade_date"].max().date().isoformat() if len(valid) else "",
        "valid_close_rows": int(len(valid)),
        "duplicate_dates": int(frame.duplicated(["trade_date"]).sum()),
        "null_counts": {field: int(frame[field].isna().sum()) for field in ALL_FIELDS},
        "max_calendar_gap_days": int(gaps.max()) if len(gaps) else None,
        "return_min": float(returns.min()) if returns.notna().any() else None,
        "return_max": float(returns.max()) if returns.notna().any() else None,
        "large_abs_return_over_20pct": int((returns.abs() > 0.20).sum()),
    }


def main():
    jq = load_joinquant()
    frames = []
    source_files = [{"path": str(JQ_ARCHIVE), "sha256": sha256(JQ_ARCHIVE), "type": "zip"}]
    for code, frame in jq.items():
        if code != "932000":
            frames.append(normalize(frame, code, "joinquant"))
    if not TONG_LIAN_FILE.exists():
        raise FileNotFoundError(TONG_LIAN_FILE)
    source_files.append({"path": str(TONG_LIAN_FILE), "sha256": sha256(TONG_LIAN_FILE), "type": "csv"})
    frames.append(normalize(pd.read_csv(TONG_LIAN_FILE, dtype={"code": str}), "932000", "tonglian"))
    panel = pd.concat(frames, ignore_index=True).sort_values(["trade_date", "index_code"])
    panel.to_csv(OUT_PANEL, index=False, encoding="utf-8-sig")
    manifest = {
        "created_on": pd.Timestamp.now().isoformat(),
        "time_axis": "union_of_valid_index_dates",
        "macro_sentiment_start": "2015-01-01",
        "source_files": source_files,
        "indices": {code: {**meta, "audit": audit_frame(panel.loc[panel["index_code"].eq(code)])} for code, meta in INDEX_META.items()},
        "panel": {"path": str(OUT_PANEL), "rows": int(len(panel)), "date_min": panel.trade_date.min().date().isoformat(), "date_max": panel.trade_date.max().date().isoformat()},
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import io
import json
import zipfile
from datetime import date, timedelta

import pandas as pd


OUTPUT_DAILY = "four_indices_daily_raw.zip"
OUTPUT_MINUTE = "four_indices_1m_raw.zip"
OUTPUT_MANIFEST = "four_indices_download_manifest.json"
END_DATE = date.today()
MINUTE_DAYS_PER_REQUEST = 365
FIELDS = ["open", "high", "low", "close", "volume", "money"]
INDICES = {
    "000852_XSHG_csi1000": {"jq_code": "000852.XSHG", "start": date(2004, 12, 31)},
    "000905_XSHG_csi500": {"jq_code": "000905.XSHG", "start": date(2004, 12, 31)},
    "000688_XSHG_star50": {"jq_code": "000688.XSHG", "start": date(2019, 12, 31)},
    "399006_XSHE_chinext": {"jq_code": "399006.XSHE", "start": date(2010, 5, 31)},
}


def as_csv(frame: pd.DataFrame) -> bytes:
    result = frame.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        result.index.name = result.index.name or "time"
        result = result.reset_index()
    return result.to_csv(index=False).encode("utf-8-sig")


def query(code: str, start: date, end: date, frequency: str) -> pd.DataFrame:
    return get_price(
        code,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        frequency=frequency,
        fields=FIELDS,
        skip_paused=False,
        panel=False,
        fq=None,
    )


manifest = {
    "run_date": END_DATE.isoformat(),
    "adjustment": "raw_index_points_fq_none",
    "note": "Index levels do not have qfq/hfq. Use a total-return index for dividend-reinvested performance.",
    "minute_days_per_request": MINUTE_DAYS_PER_REQUEST,
    "indices": {},
}

with zipfile.ZipFile(OUTPUT_DAILY, "w", compression=zipfile.ZIP_DEFLATED) as daily_zip, \
        zipfile.ZipFile(OUTPUT_MINUTE, "w", compression=zipfile.ZIP_DEFLATED) as minute_zip:
    for name, spec in INDICES.items():
        code = spec["jq_code"]
        start = spec["start"]
        status = {"jq_code": code, "daily": {}, "minute": {"chunks": [], "rows": 0, "errors": []}}

        try:
            daily = query(code, start, END_DATE, "daily")
            daily_zip.writestr(name + ".csv", as_csv(daily))
            status["daily"] = {"rows": int(len(daily)), "start": start.isoformat(), "end": END_DATE.isoformat()}
        except Exception as exc:
            status["daily"] = {"error": repr(exc)}

        current = start
        parts = []
        while current <= END_DATE:
            chunk_end = min(current + timedelta(days=MINUTE_DAYS_PER_REQUEST - 1), END_DATE)
            try:
                frame = query(code, current, chunk_end, "1m")
                if frame is not None and not frame.empty:
                    parts.append(frame)
                    status["minute"]["rows"] += int(len(frame))
                status["minute"]["chunks"].append({"start": current.isoformat(), "end": chunk_end.isoformat(), "rows": int(len(frame)) if frame is not None else 0})
            except Exception as exc:
                status["minute"]["errors"].append({"start": current.isoformat(), "end": chunk_end.isoformat(), "error": repr(exc)})
            current = chunk_end + timedelta(days=1)

        if parts:
            minute = pd.concat(parts, axis=0)
            minute = minute[~minute.index.duplicated(keep="last")]
        else:
            minute = pd.DataFrame(columns=FIELDS)
        minute_zip.writestr(name + ".csv", as_csv(minute))
        manifest["indices"][name] = status

with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)

print(json.dumps(manifest, ensure_ascii=False, indent=2))

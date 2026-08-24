import json
import zipfile
from datetime import datetime

import pandas as pd


with open("index_minute_batch.json", "r", encoding="utf-8-sig") as handle:
    config = json.load(handle)

code = config["jq_code"]
label = config["label"]
year = int(config["year"])
month = int(config["month"])
tag = "{}_{}{:02d}".format(label, year, month)
output = "{}_1m.zip".format(tag)
manifest = "{}_1m_manifest.json".format(tag)
progress = "{}_1m_progress.jsonl".format(tag)


def log_event(level, step, **fields):
    event = {"time": datetime.now().isoformat(), "level": level, "step": step}
    event.update(fields)
    with open(progress, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False))

start_date = "{}-{:02d}-01".format(year, month)
end_date = "{}-{:02d}-31".format(year, month)
log_event("INFO", "batch_started", jq_code=code, year=year, month=month)
try:
    log_event("INFO", "get_price_started")
    frame = get_price(code, start_date=start_date, end_date=end_date, frequency="1m", fields=["open", "high", "low", "close", "volume", "money"], skip_paused=False, panel=False, fq=None)
    log_event("INFO", "get_price_finished", rows=int(len(frame)))
except Exception as exc:
    log_event("ERROR", "get_price_failed", error=repr(exc))
    raise
if isinstance(frame.index, pd.DatetimeIndex):
    frame.index.name = "time"
    frame = frame.reset_index()
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    log_event("INFO", "zip_write_started")
    archive.writestr("{}.csv".format(label), frame.to_csv(index=False).encode("utf-8-sig"))
log_event("INFO", "zip_write_finished", artifact=output)
result = {"jq_code": code, "year": year, "month": month, "rows": int(len(frame)), "columns": list(frame.columns), "artifact": output}
with open(manifest, "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
log_event("INFO", "batch_finished", manifest=manifest)
print(json.dumps(result, ensure_ascii=False))

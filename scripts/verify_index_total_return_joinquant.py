import json
from datetime import date

import pandas as pd


OUTPUT = "five_indices_total_return_verification.json"
KEYWORDS = ["中证500", "中证1000", "中证2000", "科创50", "创业板指"]


def serializable_frame(frame):
    return frame.reset_index().astype(str).to_dict(orient="records")


all_indices = get_all_securities(types=["index"], date=date.today())
name_column = "display_name" if "display_name" in all_indices.columns else "name"
names = all_indices[name_column].fillna("").astype(str)
mask = pd.Series(False, index=all_indices.index)
for keyword in KEYWORDS:
    mask = mask | names.str.contains(keyword, regex=False)
matches = all_indices.loc[mask]

results = {
    "universe_date": date.today().isoformat(),
    "columns": list(all_indices.columns),
    "matching_indices": serializable_frame(matches),
    "total_return_price_probes": [],
}

for code, row in matches.iterrows():
    name = str(row.get(name_column, ""))
    if "全收益" not in name and "全收益" not in str(row.get("name", "")):
        continue
    probe = {"code": str(code), "name": name}
    try:
        data = get_price(
            str(code),
            start_date="2026-08-01",
            end_date="2026-08-14",
            frequency="daily",
            fields=["close"],
            skip_paused=False,
            panel=False,
            fq=None,
        )
        probe["status"] = "ok"
        probe["rows"] = int(len(data))
        probe["first_close"] = float(data["close"].iloc[0]) if len(data) else None
        probe["last_close"] = float(data["close"].iloc[-1]) if len(data) else None
    except Exception as exc:
        probe["status"] = "error"
        probe["error"] = repr(exc)
    results["total_return_price_probes"].append(probe)

with open(OUTPUT, "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2)

print(json.dumps(results, ensure_ascii=False, indent=2))

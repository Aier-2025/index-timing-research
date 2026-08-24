"""Download index daily OHLC data from the read-only Tonglian database."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pymysql


DEFAULT_CREDENTIALS = Path(r"G:\data\TonglianData\通联数据.txt")
DEFAULT_OUTPUT = Path(r"F:\data\index_timing_raw\932000_中证2000_daily_tonglian.csv")
START_DATE = date(2023, 8, 11)
END_DATE = date.today()


def parse_credentials(path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(host|user|pwd)\s*:\s*(.+?)\s*$", line, flags=re.I)
        if match:
            values[match.group(1).lower()] = match.group(2)
    host, port = values["host"].rsplit(":", 1)
    return {"host": host, "port": int(port), "user": values["user"], "password": values["pwd"]}


def query_index(connection, code):
    sql = """
        SELECT TICKER_SYMBOL, EXCHANGE_CD, TRADE_DATE,
               PRE_CLOSE_INDEX, OPEN_INDEX, HIGHEST_INDEX, LOWEST_INDEX,
               CLOSE_INDEX, TURNOVER_VOL, TURNOVER_VALUE, INDEX_TYPE, SOURCE
        FROM mkt_idxd_csi
        WHERE TICKER_SYMBOL = %s
          AND TRADE_DATE BETWEEN %s AND %s
        ORDER BY TRADE_DATE
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, (code, START_DATE, END_DATE))
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def normalize(frame):
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "code", "source", "open", "high", "low", "close", "volume", "money"])
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["TRADE_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["code"] = result["TICKER_SYMBOL"].astype(str)
    result["source"] = "TonglianData.mkt_idxd_csi"
    result = result.rename(
        columns={
            "OPEN_INDEX": "open",
            "HIGHEST_INDEX": "high",
            "LOWEST_INDEX": "low",
            "CLOSE_INDEX": "close",
            "TURNOVER_VOL": "volume",
            "TURNOVER_VALUE": "money",
        }
    )
    output = result[["trade_date", "code", "source", "open", "high", "low", "close", "volume", "money"]].copy()
    for column in ["open", "high", "low", "close", "volume", "money"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last").sort_values("trade_date")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code", default="932000")
    args = parser.parse_args()

    connection = pymysql.connect(database="TonglianData", charset="utf8mb4", connect_timeout=30, **parse_credentials(args.credentials_file))
    try:
        raw = query_index(connection, args.code)
    finally:
        connection.close()
    result = normalize(raw)
    if result.empty or result["close"].notna().sum() == 0:
        raise RuntimeError("Tonglian returned no valid close values for " + args.code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    manifest = {
        "source": "TonglianData.mkt_idxd_csi",
        "code": args.code,
        "requested_start": START_DATE.isoformat(),
        "requested_end": END_DATE.isoformat(),
        "rows": int(len(result)),
        "first_date": str(result["trade_date"].iloc[0]),
        "last_date": str(result["trade_date"].iloc[-1]),
        "null_counts": {column: int(result[column].isna().sum()) for column in ["open", "high", "low", "close", "volume", "money"]},
        "output": str(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

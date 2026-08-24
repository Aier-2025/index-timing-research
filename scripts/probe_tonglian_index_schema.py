"""Read-only probe for Tonglian index metadata and daily market tables."""

import json
import re
from pathlib import Path

import pymysql


CREDENTIALS = Path(r"G:\data\TonglianData\通联数据.txt")
TABLE_CANDIDATES = ["idx", "mkt_idxd_csi", "mkt_idxd", "mkt_idxd_cni", "mkt_idx_ret"]


def parse_credentials(path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*(host|user|pwd)\s*:\s*(.+?)\s*$", line, flags=re.I)
        if match:
            values[match.group(1).lower()] = match.group(2)
    host, port = values["host"].rsplit(":", 1)
    return {"host": host, "port": int(port), "user": values["user"], "password": values["pwd"]}


def main():
    connection = pymysql.connect(database="TonglianData", charset="utf8mb4", connect_timeout=30, **parse_credentials(CREDENTIALS))
    result = {"tables": {}, "index_matches": {}}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            for table in TABLE_CANDIDATES:
                if table not in tables:
                    result["tables"][table] = {"status": "missing"}
                    continue
                cursor.execute("SHOW COLUMNS FROM `" + table + "`")
                result["tables"][table] = {
                    "status": "ok",
                    "columns": [row[0] for row in cursor.fetchall()],
                }

            for table in ["idx", "mkt_idxd_csi", "mkt_idxd", "mkt_idxd_cni", "mkt_idx_ret"]:
                if table not in tables:
                    continue
                cursor.execute("SHOW COLUMNS FROM `" + table + "`")
                columns = {row[0] for row in cursor.fetchall()}
                result["index_matches"][table] = {"columns": sorted(columns)}

            if "idx" in tables:
                columns = set(result["index_matches"].get("idx", {}).get("columns", []))
                code_col = next((c for c in ["TICKER_SYMBOL", "INDEX_CODE", "SECURITY_CODE"] if c in columns), None)
                if code_col:
                    cursor.execute(
                        "SELECT DISTINCT `" + code_col + "` FROM `idx` "
                        "WHERE `" + code_col + "` IN ('932000', '932000.CSI', '000852', '000905') LIMIT 20"
                    )
                    result["index_matches"]["idx"]["code_column"] = code_col
                    result["index_matches"]["idx"]["matched_codes"] = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

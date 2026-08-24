import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def load_runner(path):
    spec = importlib.util.spec_from_file_location("jq_runner", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--remote-log", required=True)
parser.add_argument("--local-log", required=True)
parser.add_argument("--state", required=True)
parser.add_argument("--interval", type=int, default=10)
parser.add_argument("--timeout", type=int, default=1800)
args = parser.parse_args()

config_path = Path(args.config)
mod = load_runner(config_path.parent / "joinquant_auto_pipeline_runner.py")
client = mod.JoinQuantClient(mod.load_config(config_path), [])
client.ensure_login()
local_log = Path(args.local_log)
state_path = Path(args.state)
local_log.parent.mkdir(parents=True, exist_ok=True)
previous = ""
start = time.time()

while time.time() - start < args.timeout:
    state = {"status": "running", "remote_log": args.remote_log, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        response = client.session.get(client._url("/api/contents/" + mod.quote_jq_path(args.remote_log)), headers=client._xsrf_headers(), timeout=30)
        if response.status_code == 200:
            content = response.json().get("content", "")
            if content != previous:
                local_log.write_text(content, encoding="utf-8")
                previous = content
                state["bytes"] = len(content.encode("utf-8"))
                state["lines"] = len(content.splitlines())
    except Exception as exc:
        state["last_poll_error"] = repr(exc)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    if '"step": "batch_finished"' in previous or '"level": "ERROR"' in previous:
        state["status"] = "finished"
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        break
    time.sleep(args.interval)

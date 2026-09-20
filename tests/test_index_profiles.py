import json
from pathlib import Path


def test_profiles_are_differentiated():
    path = Path(__file__).resolve().parents[1] / "config" / "index_research_profiles.json"
    profiles = json.loads(path.read_text(encoding="utf-8"))
    assert profiles["000905"]["horizons_days"] != profiles["932000"]["horizons_days"]
    assert profiles["000688"]["slow_expert"]["enabled"] is False
    assert profiles["932000"]["max_candidates_per_horizon"] < profiles["000905"]["max_candidates_per_horizon"]

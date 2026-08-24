import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_uses_union_and_delays_macro_factors():
    config = json.loads((ROOT / "config" / "index_universe.json").read_text(encoding="utf-8"))
    assert config["time_axis"] == "union_of_valid_index_dates"
    assert config["macro_sentiment_start"] == "2015-01-01"
    assert config["signal_execution"] == "next_trading_day_open"
    assert len(config["indices"]) == 5


def test_download_script_has_no_future_date_fill():
    text = (ROOT / "scripts" / "build_daily_research_panel.py").read_text(encoding="utf-8")
    assert "2015-01-01" in text
    assert "shift(-1)" in text
    assert "ffill" not in text

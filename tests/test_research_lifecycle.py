import json
from pathlib import Path


def test_backtest_phase_is_one_time_and_separate_from_training():
    path = Path(__file__).resolve().parents[1] / "config" / "research_lifecycle.json"
    lifecycle = json.loads(path.read_text(encoding="utf-8"))
    backtest_phase = lifecycle["phases"]["backtest"]
    assert backtest_phase["can_change_strategy"] is False
    assert backtest_phase["max_runs_per_strategy"] == 1
    assert lifecycle["backtest_policy"]["failed_backtest_requires_new_strategy_cycle"] is True

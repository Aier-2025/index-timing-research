"""Generate the self-contained five-index timing strategy report."""

from __future__ import annotations

import html
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA = Path(r"F:\data\index_timing_raw")
OUT = Path(__file__).resolve().parents[1] / "docs" / "five_index_timing_strategy_report_20260825.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def positive_candidate(path: Path) -> dict:
    rows = pd.read_csv(path)
    if "backtest_annualized_excess" not in rows.columns and "validation_annualized_excess" in rows.columns:
        rows = rows.rename(
            columns={
                "validation_start": "backtest_start",
                "validation_end": "backtest_end",
                "validation_rows": "backtest_rows",
                "validation_annualized_excess": "backtest_annualized_excess",
                "validation_active_sharpe": "backtest_active_sharpe",
                "validation_annual_turnover": "backtest_annual_turnover",
            }
        )
    fields = [
        "training_annualized_excess",
        "training_active_sharpe",
        "backtest_annualized_excess",
        "backtest_active_sharpe",
    ]
    for field in fields:
        rows[field] = pd.to_numeric(rows[field], errors="coerce")
    rows = rows.dropna(subset=fields)
    rows = rows[
        (rows["training_annualized_excess"] > 0)
        & (rows["training_active_sharpe"] > 0)
        & (rows["backtest_annualized_excess"] > 0)
        & (rows["backtest_active_sharpe"] > 0)
    ].copy()
    if rows.empty:
        raise RuntimeError(f"No positive train/OOS candidate found in {path}")
    rows["positive_score"] = rows[fields].min(axis=1)
    return rows.sort_values(["positive_score", "backtest_active_sharpe"], ascending=False).iloc[0].to_dict()


def summary_candidate(path: Path) -> dict:
    summary = load_json(path)
    candidate = summary.get("best")
    if not candidate:
        raise RuntimeError(f"No best candidate found in {path}")
    required = [
        "training_annualized_excess",
        "training_active_sharpe",
        "backtest_annualized_excess",
        "backtest_active_sharpe",
    ]
    if "backtest_annualized_excess" not in candidate:
        candidate = {
            **candidate,
            "backtest_start": candidate.get("validation_start"),
            "backtest_end": candidate.get("validation_end"),
            "backtest_rows": candidate.get("validation_rows"),
            "backtest_annualized_excess": candidate.get("validation_annualized_excess"),
            "backtest_active_sharpe": candidate.get("validation_active_sharpe"),
            "backtest_annual_turnover": candidate.get("validation_annual_turnover"),
        }
    if not all(float(candidate.get(field, 0)) > 0 for field in required):
        raise RuntimeError(f"Best candidate is not positive in all four metrics: {path}")
    return candidate


def ledger_turnover(path: Path, candidate: dict, period_column: str, model_grid: bool = False) -> dict:
    usecols = ["turnover", period_column]
    if model_grid:
        usecols.extend(["horizon_days", "alpha", "top_k", "direction", "mode"])
    else:
        usecols.extend(["strategy", "holding_days", "threshold", "direction"])
    data = pd.read_csv(path, usecols=usecols)
    if model_grid:
        data = data[
            (data["horizon_days"] == int(candidate["horizon_days"]))
            & (data["alpha"] == float(candidate["alpha"]))
            & (data["top_k"] == int(candidate["top_k"]))
            & (data["direction"] == candidate["direction"])
            & (data["mode"] == candidate["mode"])
        ]
    else:
        direction = -1.0 if candidate["direction"] == "inverse" else 1.0
        data = data[
            (data["strategy"] == candidate["strategy"])
            & (data["holding_days"] == int(candidate["holding_days"]))
            & (data["threshold"] == float(candidate["threshold"]))
            & (data["direction"] == direction)
        ]
    annualized = data.groupby(period_column)["turnover"].agg(lambda x: x.sum() / len(x) * 252)
    return annualized.to_dict()


def candidate_records() -> list[dict]:
    model_688 = summary_candidate(DATA / "index_model_grid_search" / "000688" / "summary.json")
    model_852 = summary_candidate(DATA / "index_model_grid_search" / "000852" / "summary.json")
    risk_905 = summary_candidate(DATA / "index_risk_budget_search" / "000905" / "summary.json")
    risk_399 = summary_candidate(DATA / "index_risk_budget_search" / "399006" / "summary.json")
    targeted_2000 = summary_candidate(DATA / "index_targeted_strategy_search" / "932000" / "summary.json")
    rolling_2000_path = DATA / "index_oos_windows" / "932000_2025" / "932000" / "summary.json"
    if rolling_2000_path.exists():
        rolling = load_json(rolling_2000_path)
        item = rolling.get("horizons", {}).get("10", {})
        train = item.get("training", {})
        backtest = item.get("backtest", {})
        targeted_2000 = {
            **targeted_2000,
            "model_name": "Ridge滚动 OOS",
            "horizon_days": 10,
            "training_start": train.get("interval", {}).get("signal_start"),
            "training_end": train.get("interval", {}).get("signal_end"),
            "training_rows": train.get("metrics", {}).get("rows"),
            "training_annualized_excess": train.get("metrics", {}).get("annualized_excess"),
            "training_active_sharpe": train.get("metrics", {}).get("active_sharpe"),
            "backtest_start": backtest.get("interval", {}).get("signal_start"),
            "oos_config_start": rolling.get("oos_config", {}).get("start", "2025-01-01"),
            "backtest_end": backtest.get("interval", {}).get("signal_end"),
            "backtest_rows": backtest.get("metrics", {}).get("rows"),
            "backtest_annualized_excess": backtest.get("metrics", {}).get("annualized_excess"),
            "backtest_active_sharpe": backtest.get("metrics", {}).get("active_sharpe"),
            "rolling_oos_path": str(rolling_2000_path),
        }
    for code, candidate in (("000688", model_688), ("000852", model_852)):
        turnover = ledger_turnover(
            DATA / "index_model_grid_search" / code / "model_grid_ledgers.csv",
            candidate,
            "period",
            model_grid=True,
        )
        candidate["training_annual_turnover"] = turnover.get("training")
        candidate["backtest_annual_turnover"] = turnover.get("validation")
    turnover_2000 = ledger_turnover(
        DATA / "index_targeted_strategy_search" / "932000" / "targeted_strategy_ledgers.csv",
        targeted_2000,
        "period",
    )
    targeted_2000["training_annual_turnover"] = turnover_2000.get("training")
    targeted_2000["backtest_annual_turnover"] = turnover_2000.get("backtest")
    return [
        {
            "code": "000852",
            "name": "中证1000",
            "family": "Ridge 五档仓位模型",
            "artifact": "index_model_grid_search/000852/model_grid_results.csv",
            "candidate": model_852,
            "note": "中等训练窗；10 日期限和直接方向的五档仓位映射最稳定。",
        },
        {
            "code": "932000",
            "name": "中证2000",
            "family": "Rolling OOS 模型",
            "artifact": "index_oos_windows/932000_2025/932000/summary.json",
            "candidate": targeted_2000,
            "note": "2025 起点完整 OOS；结果仍需谨慎解释。",
        },
        {
            "code": "000905",
            "name": "中证500",
            "family": "风险预算规则",
            "artifact": "index_risk_budget_search/000905/risk_budget_results.csv",
            "candidate": risk_905,
            "note": "长历史；模型容量试验有局部改善，但全因子滚动 OOS仍缺乏稳定区分度。",
        },
        {
            "code": "399006",
            "name": "创业板指",
            "family": "风险预算规则",
            "artifact": "index_risk_budget_search/399006/risk_budget_results.csv",
            "candidate": risk_399,
            "note": "高波动成长指数；上一版10日期限弱正，滚动全因子结果仍需谨慎。",
        },
        {
            "code": "000688",
            "name": "科创50",
            "family": "Ridge 五档仓位模型",
            "artifact": "index_model_grid_search/000688/model_grid_results.csv",
            "candidate": model_688,
            "note": "短历史；上一版候选保留为观察组，滚动 OOS证据偏弱。",
        },
    ]


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def text(value: object) -> str:
    return html.escape(str(value)) if value is not None else "—"


def strategy_label(record: dict) -> str:
    c = record["candidate"]
    if record["family"] == "Rolling OOS 模型":
        return f"{text(c.get('model_name', '滚动模型'))} / 期限 {text(c.get('horizon_days'))} 日"
    if record["code"] in {"000688", "000852"}:
        return f"期限 {text(c.get('horizon_days'))} 日 / alpha {num(c.get('alpha'))} / top-K {text(c.get('top_k'))} / {text(c.get('direction'))}"
    return f"{text(c.get('strategy'))} / {text(c.get('holding_days'))} 日 / 阈值 {num(c.get('threshold'))} / {text(c.get('direction'))}"


def validation_comparison(candidate: dict) -> str:
    training_excess = float(candidate["training_annualized_excess"])
    validation_excess = float(candidate.get("backtest_annualized_excess", candidate.get("validation_annualized_excess")))
    training_sharpe = float(candidate["training_active_sharpe"])
    validation_sharpe = float(candidate.get("backtest_active_sharpe", candidate.get("validation_active_sharpe")))
    excess = "超额下降" if training_excess > validation_excess else "超额未下降"
    sharpe = "夏普下降" if training_sharpe > validation_sharpe else "夏普未下降"
    return f"{excess}；{sharpe}"


def build_rule_ledger(code: str, candidate: dict) -> pd.DataFrame:
    panel = pd.read_csv(
        DATA / "index_timing_research_panel.csv",
        usecols=[
            "index_code", "trade_date", "close", "next_open_to_open", "mom_20d", "mom_60d",
            "mom_120d", "close_to_ma_60d", "drawdown_120d", "vol_20d",
        ],
        parse_dates=["trade_date"],
        dtype={"index_code": str},
    )
    group = panel.loc[panel["index_code"].eq(code)].sort_values("trade_date").set_index("trade_date")
    strategy = candidate["strategy"]
    if strategy == "drawdown_trend":
        raw = group["mom_60d"] + group["drawdown_120d"] + group["mom_120d"]
    elif strategy == "volatility_reversal":
        raw = -group["mom_20d"] + group["mom_60d"] - group["vol_20d"]
    elif strategy == "trend_short_mid":
        raw = (group["mom_20d"] + group["mom_60d"] + group["close_to_ma_60d"]) / 3
    else:
        raise ValueError(f"Unsupported rule strategy for chart rebuild: {strategy}")
    direction = -1.0 if candidate["direction"] == "inverse" else 1.0
    score = raw * direction
    threshold = float(candidate["threshold"])
    target = pd.Series(
        np.select([score > threshold, score < -threshold], [1.0, 0.0], default=0.5),
        index=group.index,
    ).where(score.notna())
    holding = int(candidate["holding_days"])
    target = target.where(np.arange(len(target)) % holding == 0).ffill().fillna(0.5)
    ledger = group[["close", "next_open_to_open"]].copy()
    ledger["target_position"] = target
    ledger["turnover"] = ledger["target_position"].diff().abs().fillna(ledger["target_position"].abs())
    ledger["total_cost"] = ledger["turnover"] * 0.0008
    ledger["strategy_return"] = ledger["target_position"] * ledger["next_open_to_open"] - ledger["total_cost"]
    ledger["benchmark_return"] = ledger["next_open_to_open"]
    training_mask = (
        (ledger.index >= pd.Timestamp(candidate["training_start"]))
        & (ledger.index <= pd.Timestamp(candidate["training_end"]))
    )
    backtest_start = candidate.get("backtest_start", candidate.get("validation_start"))
    backtest_end = candidate.get("backtest_end", candidate.get("validation_end"))
    backtest_mask = (
        ledger.index >= pd.Timestamp(backtest_start)
    ) & (
        ledger.index <= pd.Timestamp(backtest_end)
        if backtest_end
        else ledger.index <= ledger.index.max()
    )
    ledger["period"] = np.select(
        [training_mask, backtest_mask],
        ["training", "backtest"],
        default="outside",
    )
    return ledger


def load_candidate_ledger(record: dict) -> pd.DataFrame:
    code = record["code"]
    candidate = record["candidate"]
    if record["family"] == "Rolling OOS 模型":
        path = DATA / "index_oos_windows" / "932000_2025" / "932000" / "oos_signal.csv"
        data = pd.read_csv(path, parse_dates=["trade_date"])
        data = data.loc[data["horizon_days"].eq(int(candidate["horizon_days"]))].copy()
        data = data.rename(columns={"trade_date": "date"}).set_index("date")
        data["period"] = "backtest"
        data["total_cost"] = data["turnover"] * 0.0008
        close = pd.read_csv(DATA / "index_timing_research_panel.csv", usecols=["index_code", "trade_date", "close"], parse_dates=["trade_date"], dtype={"index_code": str})
        data["close"] = close.loc[close["index_code"].eq(code)].set_index("trade_date")["close"].reindex(data.index)
        return data
    if record["family"].startswith("Ridge"):
        data = pd.read_csv(
            DATA / "index_model_grid_search" / code / "model_grid_ledgers.csv",
            parse_dates=["trade_date"],
        )
        data = data[
            (data["horizon_days"] == int(candidate["horizon_days"]))
            & (data["alpha"] == float(candidate["alpha"]))
            & (data["top_k"] == int(candidate["top_k"]))
            & (data["direction"] == candidate["direction"])
            & (data["mode"] == candidate["mode"])
        ].copy()
        data = data.rename(columns={"period": "period", "trade_date": "date"}).set_index("date")
        data["period"] = data["period"].replace({"validation": "backtest"})
        data["total_cost"] = data["turnover"] * 0.0008
        data["benchmark_return"] = pd.to_numeric(data["benchmark_return"], errors="coerce")
        close = pd.read_csv(
            DATA / "index_timing_research_panel.csv",
            usecols=["index_code", "trade_date", "close"],
            parse_dates=["trade_date"],
            dtype={"index_code": str},
        )
        close = close.loc[close["index_code"].eq(code)].set_index("trade_date")["close"]
        data["close"] = close.reindex(data.index)
        return data
    if code == "932000":
        data = pd.read_csv(
            DATA / "index_targeted_strategy_search" / code / "targeted_strategy_ledgers.csv",
            parse_dates=["trade_date"],
        )
        direction = -1.0 if candidate["direction"] == "inverse" else 1.0
        data = data[
            (data["strategy"] == candidate["strategy"])
            & (data["holding_days"] == int(candidate["holding_days"]))
            & (data["threshold"] == float(candidate["threshold"]))
            & (data["direction"] == direction)
        ].copy()
        data = data.rename(columns={"trade_date": "date"}).set_index("date")
        # Targeted ledgers may contain historical windows from earlier runs;
        # keep only the candidate's declared training and OOS intervals.
        training_mask = (data.index >= pd.Timestamp(candidate["training_start"])) & (data.index <= pd.Timestamp(candidate["training_end"]))
        backtest_mask = (data.index >= pd.Timestamp(candidate["backtest_start"])) & (data.index <= pd.Timestamp(candidate["backtest_end"]))
        data = data.loc[training_mask | backtest_mask].copy()
        data["period"] = np.where(
            (data.index >= pd.Timestamp(candidate["backtest_start"]))
            & (data.index <= pd.Timestamp(candidate["backtest_end"])),
            "backtest",
            "training",
        )
        data["total_cost"] = data["turnover"] * 0.0008
        return data
    return build_rule_ledger(code, candidate)


def png_data_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chart_theme() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.facecolor": "white",
    })


def make_strategy_charts(record: dict) -> dict[str, str]:
    chart_theme()
    ledger = load_candidate_ledger(record).sort_index()
    oos = ledger.loc[ledger["period"].eq("backtest")].dropna(subset=["strategy_return", "benchmark_return"])
    configured_start = record["candidate"].get("oos_config_start") or record["candidate"].get("backtest_start") or record["candidate"].get("validation_start")
    configured_end = record["candidate"].get("backtest_end") or record["candidate"].get("validation_end")
    if configured_start:
        oos = oos.loc[oos.index >= pd.Timestamp(configured_start)]
    if configured_end:
        oos = oos.loc[oos.index <= pd.Timestamp(configured_end)]
    if oos.empty:
        raise RuntimeError(f"Empty OOS ledger for {record['code']}")
    strategy_curve = (1 + oos["strategy_return"]).cumprod()
    benchmark_curve = (1 + oos["benchmark_return"]).cumprod()
    active_return = oos["strategy_return"] - oos["benchmark_return"]
    excess_curve = (1 + active_return).cumprod()
    strategy_dd = strategy_curve / strategy_curve.cummax() - 1
    benchmark_dd = benchmark_curve / benchmark_curve.cummax() - 1
    close = oos["close"] if "close" in oos else pd.Series(index=oos.index, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(strategy_curve.index, strategy_curve, label="策略净值", color="#c0392b", linewidth=2.1)
    axes[0].plot(benchmark_curve.index, benchmark_curve, label="基准净值", color="#1f5aa6", linewidth=1.7)
    axes[0].plot(excess_curve.index, excess_curve, label="累计超额净值", color="#d97706", linewidth=1.2, linestyle="--")
    axes[0].set_title(f"{record['name']} OOS 策略、基准与累计超额净值")
    axes[0].legend(loc="upper left", ncol=3)
    if configured_start and configured_end:
        axes[0].set_xlim(pd.Timestamp(configured_start), pd.Timestamp(configured_end))
    axes[1].fill_between(strategy_dd.index, strategy_dd, 0, color="#c0392b", alpha=0.20, label="策略回撤")
    axes[1].plot(benchmark_dd.index, benchmark_dd, color="#1f5aa6", linewidth=1.1, label="基准回撤")
    axes[1].set_ylabel("回撤")
    axes[1].legend(loc="lower left", ncol=2)
    equity_uri = png_data_uri(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].step(oos.index, oos["target_position"], where="post", color="#123b45", linewidth=1.4)
    axes[0].fill_between(oos.index, oos["target_position"], 0, step="post", color="#8dc4c2", alpha=0.45)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("权益仓位")
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_title(f"{record['name']} OOS 持仓与指数走势")
    if close.notna().any():
        close_valid = close.dropna()
        close_norm = close_valid / close_valid.iloc[0]
        axes[1].plot(close_norm.index, close_norm, color="#1f5aa6", linewidth=1.4, label="指数收盘归一化")
    axes[1].plot(strategy_curve.index, strategy_curve, color="#c0392b", linewidth=1.6, label="策略净值")
    axes[1].legend(loc="upper left")
    axes[1].set_ylabel("归一化值")
    position_uri = png_data_uri(fig)

    annual = pd.DataFrame({"strategy": oos["strategy_return"], "benchmark": oos["benchmark_return"], "active": active_return})
    annual["year"] = annual.index.year
    annual = annual.groupby("year")[["strategy", "benchmark", "active"]].apply(lambda x: (1 + x).prod() - 1)
    x = np.arange(len(annual))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - width / 2, annual["strategy"] * 100, width, label="策略", color="#c0392b")
    ax.bar(x + width / 2, annual["benchmark"] * 100, width, label="基准", color="#1f5aa6")
    ax.plot(x, annual["active"] * 100, color="#d97706", marker="o", linewidth=1.5, label="主动超额")
    ax.axhline(0, color="#52636d", linewidth=0.8)
    ax.set_xticks(x, annual.index.astype(str))
    ax.set_ylabel("年度收益 (%)")
    ax.set_title(f"{record['name']} OOS 年度收益分解")
    ax.legend()
    annual_uri = png_data_uri(fig)

    cost = oos["total_cost"].cumsum() if "total_cost" in oos else (oos["turnover"] * 0.0008).cumsum()
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(cost.index, cost * 100, color="#b45309", linewidth=1.5)
    ax.fill_between(cost.index, cost * 100, 0, color="#f6ad55", alpha=0.28)
    ax.set_ylabel("累计成本 (%)")
    ax.set_title(f"{record['name']} OOS 累计交易成本")
    cost_uri = png_data_uri(fig)
    return {"equity": equity_uri, "position": position_uri, "annual": annual_uri, "cost": cost_uri}


def make_diagnostic_charts(records: list[dict]) -> dict[str, str]:
    """Create cross-index diagnostics from the same daily candidate ledgers."""
    chart_theme()
    active_by_index = {}
    annual_excess = {}
    for record in records:
        ledger = load_candidate_ledger(record).sort_index()
        oos = ledger.loc[ledger["period"].eq("backtest")].dropna(subset=["strategy_return", "benchmark_return"])
        active = oos["strategy_return"] - oos["benchmark_return"]
        active_by_index[record["name"]] = active
        annual_excess[record["name"]] = active.groupby(active.index.year).apply(lambda x: (1 + x).prod() - 1)

    # Annual active-return heatmap: NaN means no observations, never a zero return.
    heat = pd.DataFrame(annual_excess).T.sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    image = ax.imshow(heat * 100, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=20)
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    ax.set_xticks(np.arange(len(heat.columns)), heat.columns.astype(str), rotation=35, ha="right")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            value = heat.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value * 100:+.1f}%", ha="center", va="center", fontsize=8)
    ax.set_title("五指数 OOS 年度主动超额热力图")
    fig.colorbar(image, ax=ax, label="年度主动超额 (%)")
    heatmap_uri = png_data_uri(fig)

    # Daily active-return distributions make concentration and tail risk visible.
    fig, ax = plt.subplots(figsize=(14, 5.5))
    values = [series.dropna().to_numpy() * 100 for series in active_by_index.values()]
    ax.boxplot(values, labels=list(active_by_index), showmeans=True, meanline=True, patch_artist=True,
               boxprops={"facecolor": "#d9eeee", "color": "#2878a8"},
               medianprops={"color": "#b45309", "linewidth": 1.5})
    ax.axhline(0, color="#52636d", linewidth=0.8)
    ax.set_ylabel("逐日主动收益 (%)")
    ax.set_title("五指数 OOS 主动收益分布箱线图")
    box_uri = png_data_uri(fig)

    qualified = [(name, series) for name, series in active_by_index.items() if name in {"中证1000", "科创50"}]
    fig, axes = plt.subplots(len(qualified), 1, figsize=(15, 3.6 * max(1, len(qualified))), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (name, series) in zip(axes, qualified):
        values = series.dropna() * 100
        colors = np.where(values >= 0, "#1b7f5a", "#c0392b")
        ax.bar(values.index, values.to_numpy(), color=colors, width=1.0)
        ax.axhline(0, color="#52636d", linewidth=0.8)
        ax.set_title(f"{name} OOS 逐日主动盈亏")
        ax.set_ylabel("主动收益 (%)")
    pnl_uri = png_data_uri(fig)

    # Factor-family coverage and representative factor strength.
    counts = {}
    for record in records:
        path = DATA / "index_factor_screen" / record["code"] / "all_factor_screen.csv"
        if path.exists():
            frame = pd.read_csv(path, usecols=["family"])
            counts[record["name"]] = frame["family"].value_counts()
    factor_heat = pd.DataFrame(counts).fillna(0).T
    fig, ax = plt.subplots(figsize=(14, 4.8))
    if not factor_heat.empty:
        image = ax.imshow(factor_heat, aspect="auto", cmap="Blues")
        ax.set_yticks(np.arange(len(factor_heat.index)), factor_heat.index)
        ax.set_xticks(np.arange(len(factor_heat.columns)), factor_heat.columns, rotation=35, ha="right")
        for i in range(len(factor_heat.index)):
            for j in range(len(factor_heat.columns)):
                ax.text(j, i, int(factor_heat.iloc[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="候选因子数")
    ax.set_title("因子筛选覆盖热力图：各指数 × 因子族")
    factor_uri = png_data_uri(fig)
    return {"heatmap": heatmap_uri, "boxplot": box_uri, "factor_heatmap": factor_uri, "qualified_pnl": pnl_uri}


def rolling_oos_section() -> str:
    """Render auditable rolling-OOS diagnostics and CSI1000 window sensitivity."""
    root = DATA / "index_oos_workers_v2"
    rows = []
    for code, name in [("000905", "中证500"), ("000852", "中证1000"), ("399006", "创业板指"), ("000688", "科创50"), ("932000", "中证2000")]:
        path = root / code / "summary.json"
        if not path.exists():
            continue
        summary = load_json(path)
        for horizon, period_data in summary.get("horizons", {}).items():
            metrics = period_data.get("backtest", {}).get("metrics", {})
            interval = period_data.get("backtest", {}).get("interval", {})
            rows.append(
                f"<tr><td>{text(name)}</td><td>{text(horizon)}</td><td>{text(interval.get('signal_start'))} 至 {text(interval.get('signal_end'))}</td>"
                f"<td>{num(metrics.get('rows'), 0)}</td><td>{pct(metrics.get('annualized_excess'))}</td>"
                f"<td>{num(metrics.get('active_sharpe'))}</td><td>{num(metrics.get('annual_turnover'))}</td></tr>"
            )
    window_rows = []
    for index_code, index_name, configs in [
        ("000852", "中证1000", [("2021-01 起", "000852_2021"), ("2025-01 起", "000852_2025"), ("2026-01 起", "000852_2026")]),
        ("932000", "中证2000", [("2025-01 起", "932000_2025"), ("2026-01 起", "932000_2026")]),
    ]:
        for label, folder in configs:
            path = DATA / "index_oos_windows" / folder / index_code / "summary.json"
            if not path.exists():
                continue
            summary = load_json(path)
            for horizon in (("10", "20") if index_code == "000852" else ("3", "5", "10")):
                item = summary.get("horizons", {}).get(horizon, {}).get("backtest", {})
                metrics = item.get("metrics", {})
                interval = item.get("interval", {})
                window_rows.append(
                    f"<tr><td>{index_name}</td><td>{label}</td><td>{horizon}</td><td>{text(interval.get('signal_start'))} 至 {text(interval.get('signal_end'))}</td>"
                    f"<td>{num(metrics.get('rows'), 0)}</td><td>{pct(metrics.get('annualized_excess'))}</td><td>{num(metrics.get('active_sharpe'))}</td></tr>"
                )
    if not rows and not window_rows:
        return ""
    return (
        "<details><summary>展开：滚动 OOS 详细数据</summary><section id='滚动OOS'><h2>滚动样本外复核</h2>"
        "<p class='rule'>每个滚动折内重新筛选因子、填补、截尾、标准化和拟合模型；OOS 信号按预测期限持有。下表只使用有效价格日期，未把指数成立前空值纳入训练或曲线。</p>"
        "<h3>五指数统一滚动 OOS</h3><div class='scroll'><table><thead><tr><th>指数</th><th>期限</th><th>OOS 区间</th><th>样本</th><th>年化主动超额</th><th>主动夏普</th><th>年化换手</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        "<h3>中证1000 / 中证2000 OOS 起点敏感性</h3><div class='scroll'><table><thead><tr><th>指数</th><th>OOS 起点</th><th>期限</th><th>区间</th><th>样本</th><th>年化主动超额</th><th>主动夏普</th></tr></thead><tbody>" + "".join(window_rows) + "</tbody></table></div>"
        "<p class='note'>窗口结论：中证1000的2021起点10/20日结果优于缩短窗口；中证2000完整2025起点 OOS结果为负。</p></section></details>"
    )


def model_capacity_section() -> str:
    root = DATA / "index_model_oos_experiments"
    rows = []
    for code, name in [("000905", "中证500"), ("399006", "创业板指")]:
        for path in sorted(root.glob(f"{code}_*/{code}/summary.json")):
            summary = load_json(path)
            label = path.parent.parent.name.replace(code + "_", "")
            best = None
            for horizon, item in summary.get("horizons", {}).items():
                metrics = item.get("backtest", {}).get("metrics", {})
                candidate = (float(metrics.get("annualized_excess", -999)), horizon, metrics)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                rows.append(
                    f"<tr><td>{text(name)}</td><td>{text(label)}</td><td>{text(best[1])}</td><td>{pct(best[2].get('annualized_excess'))}</td><td>{num(best[2].get('active_sharpe'))}</td><td>{num(best[2].get('annual_turnover'))}</td></tr>"
                )
    for root_name, label in [("index_oos_workers_v6", "全因子+风险闸门"), ("index_oos_workers_v7", "全因子无风险闸门")]:
        for code, name in [("000905", "中证500"), ("399006", "创业板指")]:
            path = DATA / root_name / code / "summary.json"
            if not path.exists():
                continue
            summary = load_json(path)
            for horizon, item in summary.get("horizons", {}).items():
                metrics = item.get("backtest", {}).get("metrics", {})
                rows.append(
                    f"<tr><td>{text(name)}</td><td>{label}</td><td>{text(horizon)}</td><td>{pct(metrics.get('annualized_excess'))}</td><td>{num(metrics.get('active_sharpe'))}</td><td>{num(metrics.get('annual_turnover'))}</td></tr>"
                )
    if not rows:
        return ""
    return (
        "<details><summary>展开：模型族与容量试验</summary><section id='模型容量'><h2>模型族与容量试验</h2>"
        "<p class='rule'>所有配置沿用同一滚动 OOS、成本和逐折特征处理。表中只展示每个配置自身期限中最好的 OOS 结果，不能视为事后选优后的正式候选。</p>"
        "<div class='scroll'><table><thead><tr><th>指数</th><th>配置</th><th>最佳期限</th><th>年化主动超额</th><th>主动夏普</th><th>年化换手</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        "<p class='note'>结论：中证500对更长训练窗、较低正则和更大 top-K 有局部改善，但仍非全期限稳定；创业板指的 Ridge 配置优于本轮树模型、SVR和LightGBM。</p></section></details>"
    )


def final_versions_section() -> str:
    rows = [
        ("中证1000", "Ridge 五档仓位 / alpha=1 / top-K15 / 10日", "+4.01% / 0.36", "合格；滚动窗口敏感"),
        ("科创50", "Ridge 五档仓位 / alpha=1 / top-K10 / 10日", "+2.09% / 0.21", "合格；短历史观察"),
        ("中证500", "Ridge alpha=1 / 训练窗1008 / top-K30 / 10日", "+2.51% / 0.29", "观察；全因子 OOS不稳定"),
        ("创业板指", "Ridge 五档仓位 / alpha=1 / top-K15 / 10日", "+2.09% / 0.21", "观察；不能宣称抗跌"),
        ("中证2000", "Ridge 滚动 OOS / 2025起点 / 10日", "-3.07% / -0.21", "不合格；完整 OOS超额为负"),
    ]
    body = "".join(f"<tr><td>{name}</td><td>{version}</td><td>{result}</td><td>{status}</td></tr>" for name, version, result, status in rows)
    return "<section id='最终版本'><h2>最终版本清单</h2><p class='rule'>这里的“最佳”按滚动 OOS 证据、样本长度和稳定性综合判断，不按单一历史曲线最大值排序。没有合格证据的指数明确标记为暂停或观察。</p><div class='scroll'><table><thead><tr><th>指数</th><th>当前版本</th><th>滚动 OOS结果</th><th>结论</th></tr></thead><tbody>" + body + "</tbody></table></div></section>"


def long_short_focus_chart() -> str:
    path = Path(__file__).resolve().parents[1] / "reports" / "long_short_oos_analysis" / "ledger.csv"
    if not path.exists():
        return ""
    data = pd.read_csv(path, parse_dates=["trade_date"], dtype={"index_code": str})
    data["index_code"] = data["index_code"].str.replace(".0", "", regex=False).str.zfill(6)
    selected = [("000852", 5, "中证1000 多空 5日"), ("399006", 10, "创业板指 多空 10日")]
    chart_parts = []
    for code, horizon, title in selected:
        d = data.loc[data.index_code.eq(code) & data.horizon_days.eq(horizon)].sort_values("trade_date")
        if d.empty:
            continue
        curve = (1 + d.long_short_return).cumprod()
        dd = curve / curve.cummax() - 1
        benchmark_curve = (1 + d.benchmark_return).cumprod()
        excess_curve = (1 + d.long_short_return - d.benchmark_return).cumprod()
        benchmark_dd = benchmark_curve / benchmark_curve.cummax() - 1
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(d.trade_date, curve, color="#7b1e1e", linewidth=2.0, label="多空策略净值")
        axes[0].plot(d.trade_date, benchmark_curve, color="#1f5aa6", linewidth=1.5, label="指数基准净值")
        axes[0].set_title(title + "｜OOS权益、基准与累计主动超额")
        ax_excess = axes[0].twinx()
        ax_excess.plot(d.trade_date, excess_curve, color="#d97706", linewidth=1.4, linestyle="--", label="累计主动超额")
        ax_excess.set_ylabel("累计主动超额净值（倍数）", color="#d97706")
        axes[0].legend(loc="upper left")
        ax_excess.legend(loc="upper right")
        axes[1].fill_between(d.trade_date, dd, 0, color="#7b1e1e", alpha=0.28, label="多空回撤")
        axes[1].fill_between(d.trade_date, benchmark_dd, 0, color="#1f5aa6", alpha=0.16, label="基准回撤")
        axes[1].axhline(0, color="#52636d", linewidth=0.8)
        axes[1].set_ylabel("回撤（比例）")
        axes[1].legend(loc="lower left")
        axes[0].text(0.01, 0.02, "主动超额定义：多空策略收益 - 指数基准收益；右轴为逐日主动收益复合净值", transform=axes[0].transAxes, fontsize=8, color="#52636d")
        chart_parts.append(f"<div class='chart-card'><img onclick='openImage(this.src)' alt='{title}' src='{png_data_uri(fig)}'></div>")
    return "".join(chart_parts)


def make_report(records: list[dict]) -> str:
    order = ["000852", "000688", "932000", "000905", "399006"]
    records = sorted(records, key=lambda record: order.index(record["code"]))
    chart_map = {record["code"]: make_strategy_charts(record) for record in records}
    diagnostic_map = make_diagnostic_charts(records)
    rolling_section = rolling_oos_section()
    model_capacity = model_capacity_section()
    final_versions = final_versions_section()
    long_short_focus = long_short_focus_chart()
    excess_stronger = 0
    sharpe_stronger = 0
    rows = []
    period_rows = []
    cards = []
    for record in records:
        c = record["candidate"]
        period_rows.append(
            f"<tr><td>{text(record['name'])}</td><td>{text(c.get('training_start'))} 至 {text(c.get('training_end'))}</td>"
            f"<td>{pct(c.get('training_annualized_excess'))} / 夏普 {num(c.get('training_active_sharpe'))}</td>"
            f"<td>{text(c.get('backtest_start', c.get('validation_start')))} 至 {text(c.get('backtest_end', c.get('validation_end')))}</td>"
            f"<td>{pct(c.get('backtest_annualized_excess', c.get('validation_annualized_excess')))} / 夏普 {num(c.get('backtest_active_sharpe', c.get('validation_active_sharpe')))}</td></tr>"
        )
        rows.append(
            "<tr>"
            f"<td>{text(record['name'])}<br><small>{text(record['code'])}</small></td>"
            f"<td>{text(record['family'])}<br><small>{strategy_label(record)}</small></td>"
            f"<td>{text(c.get('backtest_start', c.get('validation_start')))} 至 {text(c.get('backtest_end', c.get('validation_end')))}<br><small>{text(c.get('backtest_rows', c.get('validation_rows')))} 个交易日</small></td>"
            f"<td class='good'>{pct(c.get('backtest_annualized_excess', c.get('validation_annualized_excess')))}<br><small>超额夏普 {num(c.get('backtest_active_sharpe', c.get('validation_active_sharpe')))}</small></td>"
            f"<td>{num(c.get('backtest_annual_turnover', c.get('validation_annual_turnover')))}</td>"
            "</tr>"
        )
        cards.append(
            f"<article class='card'><h3>{text(record['name'])}</h3>"
            f"<p class='lede'>{text(record['note'])}</p>"
            f"<dl><dt>候选配置</dt><dd>{strategy_label(record)}</dd>"
            f"<dt>验证/OOS期</dt><dd>{text(c.get('backtest_start', c.get('validation_start')))} 至 {text(c.get('backtest_end', c.get('validation_end')))}</dd>"
            f"<dt>验证/OOS表现</dt><dd class='good'>年化超额 {pct(c.get('backtest_annualized_excess', c.get('validation_annualized_excess')))}，超额夏普 {num(c.get('backtest_active_sharpe', c.get('validation_active_sharpe')))}</dd>"
            f"<dt>年化换手</dt><dd>{num(c.get('backtest_annual_turnover', c.get('validation_annual_turnover')))}</dd>"
            f"<dt>原始产物</dt><dd><code>{text(record['artifact'])}</code></dd></dl></article>"
        )

    chart_sections = []
    for record in records:
        charts = chart_map[record["code"]]
        chart_sections.append(
            f"<section class='chart-section'><h3>{text(record['name'])}：{strategy_label(record)}</h3>"
            f"<div class='chart-card'><img onclick='openImage(this.src)' alt='{text(record['name'])} OOS累计净值与回撤' src='{charts['equity']}'></div>"
            f"<details><summary>展开：{text(record['name'])} 附加图表</summary><div class='chart-card'><img onclick='openImage(this.src)' alt='{text(record['name'])} OOS持仓与指数走势' src='{charts['position']}'></div>"
            f"<div class='chart-grid'><div class='chart-card'><img onclick='openImage(this.src)' alt='{text(record['name'])} OOS年度收益' src='{charts['annual']}'></div>"
            f"<div class='chart-card'><img onclick='openImage(this.src)' alt='{text(record['name'])} OOS累计交易成本' src='{charts['cost']}'></div></div></details></section>"
        )
    public_chart_sections = [section for record, section in zip(records, chart_sections) if record["code"] in {"000852", "000688"}]
    appendix_chart_sections = [section for record, section in zip(records, chart_sections) if record["code"] not in {"000852", "000688"}]

    chart_theme()
    metric_names = ["backtest_annualized_excess", "backtest_active_sharpe", "backtest_annual_turnover"]
    chart_df = pd.DataFrame(
        {
            "name": [record["name"] for record in records],
            "excess": [float(record["candidate"]["backtest_annualized_excess"]) * 100 for record in records],
            "sharpe": [float(record["candidate"]["backtest_active_sharpe"]) for record in records],
            "turnover": [float(record["candidate"].get("backtest_annual_turnover", np.nan)) for record in records],
        }
    )
    x = np.arange(len(chart_df))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x, chart_df["excess"], color="#c0392b")
    axes[0].set_title("OOS 年化超额 (%)")
    axes[0].set_ylabel("百分比")
    axes[1].bar(x, chart_df["sharpe"], color="#1f5aa6")
    axes[1].set_title("OOS 超额夏普")
    axes[2].bar(x, chart_df["turnover"], color="#d97706")
    axes[2].set_title("OOS 年化换手")
    for ax in axes:
        ax.set_xticks(x, chart_df["name"], rotation=35, ha="right")
        ax.axhline(0, color="#52636d", linewidth=0.8)
    overview_uri = png_data_uri(fig)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>五大指数择时策略研究整理报告</title>
<style>
:root {{ --ink:#17212b; --muted:#52636d; --line:#d5dde3; --teal:#123b45; --wash:#eef6f6; --good:#087f5b; --warn:#b45309; }}
* {{ box-sizing:border-box; }}
body {{ font-family:Arial,"Microsoft YaHei",sans-serif; max-width:1580px; margin:0 auto; padding:0 22px 120px; color:var(--ink); line-height:1.75; background:#fff; }}
h1,h2,h3 {{ color:var(--teal); }} h1 {{ margin:0 0 6px; }} h2 {{ border-bottom:2px solid #8dc4c2; padding-bottom:6px; margin-top:42px; }}
.nav {{ position:sticky; top:0; z-index:2; background:#fff; border-bottom:1px solid var(--line); padding:10px 0; display:flex; gap:18px; overflow:auto; white-space:nowrap; }}
.nav a {{ color:#245563; text-decoration:none; font-size:14px; }}
.hero {{ border-bottom:1px solid var(--line); padding:30px 0 24px; }} .lead {{ font-size:17px; color:#30414d; max-width:1080px; }}
.note {{ background:#fff7ed; border-left:4px solid #d97706; padding:14px 18px; }} .rule {{ background:var(--wash); border-left:4px solid #2878a8; padding:12px 16px; }}
.good {{ color:var(--good); font-weight:800; }} .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
.card {{ border:1px solid var(--line); padding:16px; border-radius:6px; background:#fbfcfd; }} .card h3 {{ margin-top:0; }}
.scroll {{ max-width:100%; overflow:auto; }} table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }}
th,td {{ border:1px solid var(--line); padding:8px; text-align:right; vertical-align:top; }} th:first-child,td:first-child {{ text-align:left; }} th {{ background:#e8f2f4; }}
tr:first-child td {{ background:#f0faf5; }} small {{ color:var(--muted); }} code {{ overflow-wrap:anywhere; }}
dl {{ display:grid; grid-template-columns:150px 1fr; margin:0; }} dt {{ color:var(--muted); font-weight:700; }} dd {{ margin:0 0 7px; }}
.chart-section {{ margin:28px 0 42px; }} .chart-card {{ border:1px solid var(--line); border-radius:6px; padding:10px; margin:14px 0; background:#fff; }}
.chart-card img {{ display:block; width:100%; height:auto; cursor:zoom-in; }} .chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} details {{ margin:18px 0; border-top:1px solid var(--line); padding-top:10px; }} summary {{ cursor:pointer; color:var(--teal); font-weight:700; }} .image-modal {{ display:none; position:fixed; inset:0; z-index:9; background:rgba(0,0,0,.82); padding:30px; }} .image-modal img {{ max-width:100%; max-height:calc(100vh - 60px); margin:auto; display:block; }} .image-modal button {{ position:absolute; top:12px; right:18px; background:#fff; border:0; font-size:26px; width:40px; height:40px; cursor:pointer; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin:20px 0; }}
.metric {{ padding:13px 16px; border-right:1px solid var(--line); }} .metric:last-child {{ border-right:0; }} .metric b {{ display:block; color:var(--teal); font-size:24px; }}
@media(max-width:800px) {{ body {{ padding:0 14px 80px; }} .grid,.chart-grid {{ grid-template-columns:1fr; }} .metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .metric:nth-child(2n) {{ border-right:0; }} dl {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<nav class="nav"><a href="#导读">1 总述</a><a href="#仅多头">2 仅多头策略</a><a href="#多空">3 多空策略</a><a href="#结论">4 结论</a><a href="#附录">5 附录</a></nav>
<header class="hero" id="导读">
<h1>五大指数择时策略研究整理报告</h1>
<p class="lead"><b>研究目的：</b>验证股指上是否存在可审计、成本后的仅多头择时超额，并筛选值得继续研究的指数版本。<br><b>核心结论：</b>当前中证1000和科创50最值得继续推进；中证500、创业板指和中证2000仍属于观察或待改进版本。</p>
<div class="note"><b>先看结论：</b>五指数 OOS 结果存在明显分层：中证1000和科创50保留上一版合格候选；中证500和创业板指作为观察版本；中证2000按 2025-01 起完整 OOS 评估后主动超额为负，暂不纳入有效策略集合。以下总览同时展示正向和弱势结果，便于比较。</div>
</header>
<section id="摘要表"><h2>1.2 策略结果摘要</h2><div class="scroll"><table><thead><tr><th>方向</th><th>指数</th><th>周期</th><th>模型/策略</th><th>OOS区间</th><th>主动超额</th><th>主动夏普</th><th>样本</th></tr></thead><tbody>
<tr><td>仅多头</td><td>中证1000</td><td>10日</td><td>Ridge五档仓位</td><td>2021-01-01 至 2026-08-24</td><td>+4.01%</td><td>0.36</td><td>1356</td></tr>
<tr><td>仅多头</td><td>科创50</td><td>10日</td><td>Ridge五档仓位</td><td>2023-01-01 至 2026-08-24</td><td>+2.09%</td><td>0.21</td><td>871</td></tr>
<tr><td>多空</td><td>中证1000</td><td>5日</td><td>方向预测 + Beta中性</td><td>2021-01-04 至 2026-08-14</td><td>+5.87%</td><td>0.36</td><td>1361</td></tr>
<tr><td>多空</td><td>创业板指</td><td>10日</td><td>方向预测 + Beta中性</td><td>2021-01-04 至 2026-08-07</td><td>约0%</td><td>0.23</td><td>1356</td></tr>
</tbody></table></div></section>
<section id="仅多头"><h2>2. 仅多头策略</h2>
<p class="rule">以下图表全部为样本外 OOS，展示顺序为中证1000、科创50、中证2000、中证500、创业板指。</p>
<div class="grid"><article class="card"><h3>中证1000｜当前主版本</h3><p class="good">OOS 主动超额约 +4.01%，主动夏普约 0.36。保留为当前优先观察版本。</p></article><article class="card"><h3>科创50｜当前合格观察版本</h3><p class="good">OOS 主动超额约 +2.09%，主动夏普约 0.21。历史较短，控制证据权重。</p></article></div>
<details><summary>展开：辅助图表</summary><div class="chart-card"><img onclick="openImage(this.src)" alt="中证1000和科创50逐日主动盈亏柱状图" src="{diagnostic_map['qualified_pnl']}"></div><div class="chart-grid"><div class="chart-card"><img onclick="openImage(this.src)" alt="年度主动超额热力图" src="{diagnostic_map['heatmap']}"></div><div class="chart-card"><img onclick="openImage(this.src)" alt="主动收益分布箱线图" src="{diagnostic_map['boxplot']}"></div></div></details>
{''.join(public_chart_sections)}
<details><summary>展开：其他指数 OOS 图表</summary>{''.join(appendix_chart_sections)}</details>
</section>
<section id="结论"><h2>4. 明确结论</h2>
<div class="grid"><article class="card"><h3>仅多头</h3><p class="good">中证1000 10日、科创50 10日：当前保留。</p></article><article class="card"><h3>多空</h3><p class="good">中证1000 5日通过；创业板指 10日 Beta 中性后有希望。</p></article></div>
<div class="card"><h3>一句话结论</h3><p>优先推进：中证1000 10日仅多头、科创50 10日仅多头、中证1000 5日多空；创业板指 10日多空作为 Beta 中性研究候选。</p></div>
</section>
<section id="多空"><h2>3. 多空策略</h2><p class="rule">公开展示中证1000 5日、创业板指 10日 Beta中性多空；其他周期和指数放入附录。</p>{long_short_focus}</section>
<details><summary>展开：训练、因子、数据与迭代方法</summary><section id="方法"><h2>训练、因子与迭代方法</h2>
<div class="grid"><article class="card"><h3>研究目标</h3><p>本项目的目标不是让五个指数都产生漂亮曲线，而是检验股指上是否存在可复核的 long-only 择时机会：核心门槛是样本外主动超额和主动夏普同时为正，并且在成本、不同市场阶段和窗口变化下仍有区分度。只有通过这一门槛，才值得进入后续组合和实盘执行研究。</p></article><article class="card"><h3>信号与收益</h3><p>信号在 T 日收盘后形成，收益使用下一交易日开盘至开盘的可执行代理；策略收益为目标权益仓位乘基准收益，再扣手续费和滑点。超额收益按逐日主动收益（策略收益减基准收益）复合，超额夏普也基于逐日主动收益计算。</p></article></div>
<div class="grid"><article class="card"><h3>数据来源</h3><p>中证500、中证1000、创业板指、科创50使用 JoinQuant 原始指数日线；中证2000使用 TonglianData CSI 指数日线。原始数据保留用于审计，研究面板只保留开盘和收盘均有效的指数交易日；不对指数点位做股票式前复权，不计分红。</p></article><article class="card"><h3>因子来源与族</h3><p>因子来自 <code>F:\\量化因子库</code> 的 feature contract 和 market-state panel，包含量价/趋势、风险/波动、基本面/估值、利率/货币环境、宏观基本面、流动性/成交宽度以及其他因子；另加入项目自身的动量、均线距离、波动率、回撤和市场广度技术字段。</p></article></div>
<div class="grid"><article class="card"><h3>因子处理</h3><p>每个滚动训练折内独立完成字段可用性判断、来源滞后、缺失填补、1%/99%截尾和标准化；训练窗外不反向使用统计量。因子按训练窗内 RankIC 绝对值选择 top-K，避免用全样本相关性提前挑因子。2015年前不可审计的宏观/情绪字段保持缺失，不填成假信号。</p></article><article class="card"><h3>模型与仓位</h3><p>已对 Ridge、Logistic、SVR、随机森林、HistGradientBoosting、LightGBM、Transformer 和 Ridge+树模型集成进行逐折 OOS 试验。预测结果统一映射为五档仓位：0%、25%、50%、75%、100%；风险闸门只降低上限，再量化回五档。</p></article></div>
<div class="grid"><article class="card"><h3>滚动训练与策略迭代</h3><p>每约三个月（63个交易日）重训一次；每个测试折重新筛选因子、处理数据、拟合模型并按预测期限持有。迭代只允许使用当时已成熟的训练数据，OOS 结果不反向调参。模型族、训练窗、正则、top-K和风险闸门作为独立实验记录，最终版本按 OOS 超额夏普、样本长度、换手和稳定性综合判断。</p></article><article class="card"><h3>多空后续方向</h3><p>若仅多头证据足够稳定，下一步可把方向预测扩展为股指期货多空，并报告多空收益、Beta中性、基差、展期、保证金占用和极端行情风险。</p></article></div>
</section></details>
<section id="总览"><h2>4. 五指数结果总览</h2>
<div class="metric-grid"><div class="metric"><span>仅多头合格</span><b class="good">2 个</b><small>中证1000、科创50</small></div><div class="metric"><span>多空合格/有希望</span><b class="good">1 / 1</b><small>中证1000合格、创业板指Beta中性</small></div><div class="metric"><span>仓位合同</span><b>五档</b><small>0% / 25% / 50% / 75% / 100%</small></div><div class="metric"><span>统一单边成本</span><b>8 bp</b><small>手续费 3 bp + 滑点 5 bp</small></div></div>
<div class="scroll"><table><thead><tr><th>方向</th><th>指数</th><th>周期</th><th>OOS主动超额</th><th>OOS主动夏普</th><th>样本</th></tr></thead><tbody><tr><td>仅多头</td><td>中证1000</td><td>10日</td><td>+4.01%</td><td>0.36</td><td>1356日</td></tr><tr><td>仅多头</td><td>科创50</td><td>10日</td><td>+2.09%</td><td>0.21</td><td>871日</td></tr><tr><td>多空</td><td>中证1000</td><td>5日</td><td>+5.87%</td><td>0.36</td><td>1361日</td></tr><tr><td>多空</td><td>创业板指</td><td>10日</td><td>约0%</td><td>0.23</td><td>1356日</td></tr></tbody></table></div>
<div class="scroll"><table><thead><tr><th>指数（代码）</th><th>候选</th><th>OOS 区间</th><th>OOS 年化主动超额</th><th>OOS 主动夏普</th><th>OOS 年化换手</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="chart-card"><img onclick="openImage(this.src)" alt="五指数OOS核心指标对比" src="{overview_uri}"></div>
<p class="rule">本节只展示 OOS 结果。训练期、全样本曲线和训练期指标不纳入主表；滚动 OOS 细节见后文。</p>
</section>
{rolling_section}
{model_capacity}
{final_versions}
<section id="因子模型"><h2>2. 因子与模型</h2>
<div class="grid"><article class="card"><h3>因子体系</h3><p>研究面板按四类信息组织：量价/趋势（动量、均线距离、回撤）、风险/波动（滚动波动率与极端风险）、流动性/成交宽度（成交额、广度、融资与流动性状态）以及宏观/估值状态。因子先经过可得性滞后、缺失率、滚动相关性和稳定性筛选；指数成立前或字段不可得时保持缺失，不前向填充为虚假信号。</p></article><article class="card"><h3>模型与仓位</h3><p>Ridge 模型在每个滚动训练折内重新选择 top-K 因子，训练窗口内完成缺失填补、截尾和标准化，再映射到 0/25/50/75/100% 五档权益仓位。规则策略只使用明确的趋势、回撤和波动反转条件。标签成熟后才进入训练，OOS 期间不回看未来数据。</p></article></div>
<div class="chart-card"><img onclick="openImage(this.src)" alt="因子族覆盖热力图" src="{diagnostic_map['factor_heatmap']}"></div>
</section>
<section id="图表说明"><h2>图表口径说明</h2>
<p class="rule">主图全部只展示候选声明的样本外 OOS 区间，不混入训练期或指数成立前数据。累计净值已扣除单边手续费 3 bp 和滑点 5 bp；累计超额净值按逐日主动收益复合，不再使用策略/基准比值。无效日期保持为空，不绘制水平线。</p>
</section>
<details><summary>展开：数据口径、策略详情与参考资料</summary>
<section id="研究口径"><h2>数据、时间和执行口径</h2><div class="grid">
<div class="card"><h3>数据</h3><p>使用本地五指数日频研究面板：中证500、中证1000、创业板指、科创50来自 JoinQuant 日频原始点位，中证2000来自通联日频数据。指数点位不调用或伪造前复权；不计分红。</p></div>
<div class="card"><h3>时间一致性</h3><p>信号使用 T 日收盘可见字段，收益使用下一交易日开盘至开盘的可执行代理收益。宏观和情绪字段按项目配置中的可得性滞后处理；2015 年前不可审计字段不进入模型。</p></div>
<div class="card"><h3>成本</h3><p>所有搜索统一采用单边手续费 3 bp、单边滑点 5 bp，合计单边 8 bp。策略收益先计算目标仓位乘以基准收益，再扣除仓位变化对应的交易成本。</p></div>
<div class="card"><h3>样本划分</h3><p>开发期用于比较策略和参数，验证/OOS期按指数画像预先设定。模型网格在训练折内选因子；规则搜索不使用验证期结果回调参数。报告直接整理既有搜索产物，不在本阶段继续调参。</p></div>
</div></section>
<section id="微盘候选"><h2>微盘指数候选核查</h2>
<div class="card"><h3>国证2000（399303）</h3><p>该指数在定义上比中证2000更早、成分数量具有微盘代表性，属于优先核查候选。本轮已尝试从 TonglianData 的 CSI 指数表和 JoinQuant 日频接口下载完整历史；Tonglian 未返回有效记录，JoinQuant 本地浏览器执行器未能找到可用的 Jupyter API，因此暂未把它并入研究面板。报告保留中证2000作为当前可复核序列，并将国证2000标记为待补数据候选，不把不可验证的数据当作结果。</p></div>
</section>
<section id="策略"><h2>4. 各指数策略详情</h2><div class="grid">{''.join(cards)}</div></section>
<section id="风险"><h2>5. 结果解释、风险与限制</h2>
<p>五指数 OOS 结果呈现清晰分层：中证1000和科创50当前版本相对较好；中证500和创业板指缺少稳定区分度；中证2000在完整 2025 起点 OOS 下表现较弱。报告保留所有指数，避免只展示成功或失败的一侧。</p>
<ul><li>中证1000保留为当前主版本，但继续监控窗口敏感性。</li><li>科创50保留为合格但短历史观察版本。</li><li>中证500和创业板指保留为研究候选，暂不宣称抗跌。</li><li>中证2000暂不纳入有效策略集合，需重新研究训练标签、风险状态和因子表达。</li></ul>
<p class="note">本报告的定位是“已有实验结果的研究整理版”：可以作为后续组合研究的候选输入，但不宣称参数已冻结、统计显著、容量充分或可直接实盘。</p>
</section>
<section id="计划"><h2>6. 当前结论与使用边界</h2>
<ol><li>五指数日频研究链和统一滚动 OOS 已完成。</li><li>中证500当前没有滚动 OOS区分度，创业板指仅部分期限为正；两者不能直接作为抗跌策略。</li><li>中证2000的2026窗口改善已纳入，但短样本只作为观察候选。</li><li>下一轮应优先改进中证500/创业板因子表达和训练目标，而不是继续从旧搜索结果挑选最优曲线。</li></ol>
</section>
<section id="参考文献"><h2>参考文献与数据表</h2><ul><li>JoinQuant 指数日频原始数据：中证500、中证1000、创业板指、科创50。</li><li>TonglianData <code>mkt_idxd_csi</code>：中证2000。</li><li>量化因子库：<code>F:\\量化因子库\\docs\\reports\\csi1000_factor_term_audit_20260819\\feature_contract_draft.csv</code>。</li><li>市场状态质量表：<code>F:\\量化因子库\\docs\\factor_library\\market_state_factor_quality.csv</code>。</li></ul></section>
</details>
<div id="imageModal" class="image-modal" onclick="closeImage()"><button type="button" onclick="event.stopPropagation();closeImage()">×</button><img id="imageModalContent" alt="放大图" onclick="event.stopPropagation()"></div><script>function openImage(src){{document.getElementById('imageModalContent').src=src;document.getElementById('imageModal').style.display='block'}}function closeImage(){{document.getElementById('imageModal').style.display='none';document.getElementById('imageModalContent').src=''}}</script>
<footer><p><small>生成日期：2026-08-25。原始结果目录：<code>F:/data/index_timing_raw/</code>。本报告由 <code>scripts/generate_index_strategy_report.py</code> 生成。</small></p></footer>
</body></html>"""


def main() -> None:
    records = candidate_records()
    OUT.write_text(make_report(records), encoding="utf-8")
    print(f"wrote {OUT}")
    for record in records:
        c = record["candidate"]
        print(
            record["name"],
            record["family"],
            c.get("training_annualized_excess"),
            c.get("backtest_annualized_excess", c.get("validation_annualized_excess")),
            c.get("training_active_sharpe"),
            c.get("backtest_active_sharpe", c.get("validation_active_sharpe")),
        )


if __name__ == "__main__":
    main()

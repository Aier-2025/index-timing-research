"""Run an isolated rolling OOS worker for one index.

Feature selection, imputation, clipping, scaling and model fitting are repeated
inside each historical training fold. No full-sample candidate file is used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

import screen_index_timing_factors as screen


DATA_DIR = Path(r"F:\data\index_timing_raw")
PANEL = DATA_DIR / "index_timing_research_panel.csv"
CONTRACT = Path(r"F:\量化因子库\docs\reports\csi1000_factor_term_audit_20260819\feature_contract_draft.csv")
STATE_QUALITY = Path(r"F:\量化因子库\docs\factor_library\market_state_factor_quality.csv")
PROFILES = Path(__file__).resolve().parents[1] / "config" / "index_research_profiles.json"
COMMISSION_ONE_WAY = 0.0003
SLIPPAGE_ONE_WAY = 0.0005
ONE_WAY_COST = COMMISSION_ONE_WAY + SLIPPAGE_ONE_WAY


def prepare(train_x, test_x):
    median = train_x.median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = train_x.replace([np.inf, -np.inf], np.nan).fillna(median)
    test = test_x.replace([np.inf, -np.inf], np.nan).fillna(median)
    lower = train.quantile(0.01)
    upper = train.quantile(0.99)
    train = train.clip(lower, upper, axis=1)
    test = test.clip(lower, upper, axis=1)
    mean = train.mean()
    scale = train.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return (train - mean) / scale, (test - mean) / scale


def fit_predict(model_name, train_x, train_y, test_x, alpha):
    """Fit one deterministic model family inside a rolling fold."""
    if model_name == "ensemble":
        members = ["ridge", "random_forest", "lightgbm"]
        predictions = [fit_predict(member, train_x, train_y, test_x, alpha) for member in members]
        train_pred = sum(item[0] for item in predictions) / len(predictions)
        test_pred = sum(item[1] for item in predictions) / len(predictions)
        return train_pred, test_pred
    if model_name == "transformer":
        import torch
        import torch.nn as nn

        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(10, max(3, len(train_x) // 20))
        train_values = train_x.to_numpy(dtype=np.float32)
        test_values = test_x.to_numpy(dtype=np.float32)
        train_target = train_y.to_numpy(dtype=np.float32)

        def sequences(values, targets=None):
            xs, ys, ends = [], [], []
            start = seq_len - 1
            for end in range(start, len(values)):
                xs.append(values[end - seq_len + 1:end + 1])
                ends.append(end)
                if targets is not None:
                    ys.append(targets[end])
            return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), ends

        train_seq, train_y_seq, train_ends = sequences(train_values, train_target)
        context = np.vstack([train_values[-(seq_len - 1):], test_values])
        test_seq, _, test_ends = sequences(context)
        class TinyTransformer(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.proj = nn.Linear(features, 32)
                layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))
            def forward(self, x):
                z = self.encoder(self.proj(x))
                return self.head(z[:, -1]).squeeze(-1)
        model = TinyTransformer(train_values.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        x_tensor = torch.from_numpy(train_seq).to(device)
        y_tensor = torch.from_numpy(train_y_seq).to(device)
        model.train()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(x_tensor), y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_values_pred = model(x_tensor).detach().cpu().numpy()
            test_values_pred = model(torch.from_numpy(test_seq).to(device)).detach().cpu().numpy()
        train_pred = np.full(len(train_x), train_values_pred[0] if len(train_values_pred) else 0.0, dtype=float)
        train_pred[train_ends] = train_values_pred
        return pd.Series(train_pred, index=train_x.index), pd.Series(test_values_pred, index=test_x.index)
    if model_name == "ridge":
        model = Ridge(alpha=float(alpha))
    elif model_name == "logistic":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=7)
        train_y = (train_y > 0).astype(int)
    elif model_name == "svr":
        model = SVR(C=1.0, epsilon=0.001, gamma="scale")
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=7, n_jobs=1)
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, random_state=7)
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.03, min_child_samples=20, reg_lambda=1.0, verbosity=-1, random_state=7, n_jobs=1)
    else:
        raise ValueError(f"unsupported model: {model_name}")
    if model_name == "lightgbm":
        safe_columns = [f"f{i}" for i in range(train_x.shape[1])]
        train_x = train_x.copy()
        test_x = test_x.copy()
        train_x.columns = safe_columns
        test_x.columns = safe_columns
    model.fit(train_x, train_y.to_numpy(float))
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    return pd.Series(train_pred, index=train_x.index), pd.Series(test_pred, index=test_x.index)


def apply_risk_overlay(position, group, train_dates, test_dates, enabled, mode="conservative"):
    """Cap exposure during observable downtrend/drawdown/high-volatility states."""
    if not enabled:
        return position
    state = group.reindex(test_dates)
    train = group.reindex(train_dates)
    vol_cut = pd.to_numeric(train.get("vol_20d"), errors="coerce").quantile(0.80)
    cap = pd.Series(1.0, index=test_dates)
    if mode == "defensive":
        downtrend = state["mom_60d"].lt(0) & state["close_to_ma_60d"].lt(0)
        deep_drawdown = state["drawdown_120d"].lt(-0.20)
    else:
        downtrend = state["mom_20d"].lt(0) & state["mom_60d"].lt(0)
        deep_drawdown = state["drawdown_120d"].lt(-0.25)
    high_vol_down = state["vol_20d"].gt(vol_cut) & state["mom_20d"].lt(0)
    cap.loc[downtrend.fillna(False)] = 0.50
    cap.loc[high_vol_down.fillna(False)] = cap.loc[high_vol_down.fillna(False)].clip(upper=0.50)
    cap.loc[deep_drawdown.fillna(False)] = 0.25
    # Keep the five-level position contract after applying the risk cap.
    return (position.clip(upper=cap) * 4.0).round() / 4.0


def metrics(ledger):
    data = ledger.dropna(subset=["strategy_return", "benchmark_return"])
    if data.empty:
        return {"rows": 0}
    ret = data["strategy_return"]
    bench = data["benchmark_return"]
    active = ret - bench
    active_curve = (1.0 + active).cumprod()
    equity = (1.0 + ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "rows": int(len(data)),
        "annualized_return": float((1.0 + ret).prod() ** (252.0 / len(ret)) - 1.0),
        "benchmark_annualized_return": float((1.0 + bench).prod() ** (252.0 / len(bench)) - 1.0),
        "annualized_excess": float(active_curve.iloc[-1] ** (252.0 / len(active)) - 1.0),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(252.0)) if ret.std(ddof=1) else None,
        "active_sharpe": float(active.mean() / active.std(ddof=1) * np.sqrt(252.0)) if active.std(ddof=1) else None,
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(data["turnover"].sum() / len(data) * 252.0),
        "total_commission": float(data["commission"].sum()),
        "total_slippage": float(data["slippage"].sum()),
        "total_cost": float(data["total_cost"].sum()),
    }


def interval_info(ledger):
    if ledger.empty:
        return {"rows": 0, "signal_start": None, "signal_end": None, "duplicate_dates": 0, "max_calendar_gap_days": None}
    date_values = ledger["trade_date"] if "trade_date" in ledger.columns else ledger.index
    dates = pd.DatetimeIndex(pd.to_datetime(date_values)).sort_values()
    gaps = dates.to_series().diff().dt.days.dropna()
    return {
        "rows": int(len(dates)),
        "signal_start": dates.min().date().isoformat(),
        "signal_end": dates.max().date().isoformat(),
        "duplicate_dates": int(dates.duplicated().sum()),
        "max_calendar_gap_days": int(gaps.max()) if len(gaps) else 0,
        "concatenated_chronological_series": True,
    }


def run_horizon(group, factor_frame, fields, horizon, profile, evaluation_start, evaluation_end, period):
    group = group.sort_index()
    labels = screen.future_return(group.reset_index(), horizon)
    labels.index = group.index
    dates = group.index[(group.index >= pd.Timestamp(evaluation_start)) & (group.index <= pd.Timestamp(evaluation_end)) & labels.notna()]
    if len(dates) == 0:
        return pd.DataFrame(), pd.DataFrame()
    factor_matrix = factor_frame.set_index("trade_date")[fields].reindex(group.index)
    rows = []
    folds = []
    train_days = int(profile["oos"]["train_days"])
    refit_days = int(profile["oos"]["refit_days"])
    top_k = int(profile["oos"]["top_k"])
    for begin in range(0, len(dates), refit_days):
        test_dates = dates[begin:begin + refit_days]
        if len(test_dates) == 0:
            continue
        first_test_pos = group.index.get_loc(test_dates[0])
        cutoff_pos = max(0, first_test_pos - horizon)
        completed = labels.iloc[:cutoff_pos].dropna().index
        train_dates = completed[-train_days:]
        if len(train_dates) < train_days:
            continue
        train_y = labels.reindex(train_dates)
        train_x = factor_matrix.reindex(train_dates)
        valid_fields = train_x.notna().sum().ge(max(60, int(train_days * 0.5))) & train_x.nunique(dropna=True).gt(1)
        train_x = train_x.loc[:, valid_fields]
        if train_x.empty:
            continue
        rank_ic = train_x.rank().corrwith(train_y.rank()).abs().sort_values(ascending=False)
        selected = rank_ic.head(top_k).index.tolist()
        train_x, test_x = prepare(train_x[selected], factor_matrix.reindex(test_dates)[selected])
        model_name = profile["oos"].get("model", "ridge")
        train_pred, test_pred = fit_predict(model_name, train_x, train_y, test_x, profile["oos"].get("alpha", 20.0))
        cuts = train_pred.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        position = pd.Series(np.select([test_pred <= cuts[0], test_pred <= cuts[1], test_pred <= cuts[2], test_pred <= cuts[3]], [0.0, 0.25, 0.5, 0.75], default=1.0), index=test_dates)
        position = position.where(np.arange(len(position)) % horizon == 0).ffill().fillna(0.5)
        position = apply_risk_overlay(position, group, train_dates, test_dates, bool(profile["oos"].get("risk_overlay", False)), profile["oos"].get("risk_mode", "conservative"))
        for day in test_dates:
            rows.append({"trade_date": day, "period": period, "model": model_name, "horizon_days": horizon, "prediction": float(test_pred.loc[day]), "target_position": float(position.loc[day]), "selected_features": ";".join(selected), "train_start": train_dates[0], "train_end": train_dates[-1]})
        folds.append({"period": period, "horizon_days": horizon, "test_start": test_dates[0], "test_end": test_dates[-1], "train_start": train_dates[0], "train_end": train_dates[-1], "train_rows": len(train_dates), "selected_feature_count": len(selected), "selected_features": ";".join(selected)})
    signal = pd.DataFrame(rows).drop_duplicates("trade_date").set_index("trade_date").sort_index() if rows else pd.DataFrame()
    return signal, pd.DataFrame(folds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, choices=["000905", "000852", "399006", "000688", "932000"])
    parser.add_argument("--output-root", type=Path, default=DATA_DIR / "index_oos_workers")
    parser.add_argument("--oos-start", help="Override the configured OOS start for window-sensitivity runs")
    parser.add_argument("--alpha", type=float, help="Override Ridge regularization")
    parser.add_argument("--train-days", type=int, help="Override rolling training window")
    parser.add_argument("--refit-days", type=int, help="Override rolling refit interval")
    parser.add_argument("--top-k", type=int, help="Override per-fold feature count")
    parser.add_argument("--model", choices=["ridge", "logistic", "svr", "random_forest", "hist_gradient_boosting", "lightgbm", "ensemble", "transformer"], default="ridge")
    parser.add_argument("--risk-mode", choices=["conservative", "defensive"], help="Risk overlay sensitivity")
    parser.add_argument("--no-risk-overlay", action="store_true", help="Disable exposure cap for factor-only comparison")
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    profile = json.loads(json.dumps(profiles[args.index]))
    if args.oos_start:
        profile["oos"]["start"] = args.oos_start
    for key, value in (("alpha", args.alpha), ("train_days", args.train_days), ("refit_days", args.refit_days), ("top_k", args.top_k)):
        if value is not None:
            profile["oos"][key] = value
    profile["oos"]["model"] = args.model
    if args.risk_mode:
        profile["oos"]["risk_mode"] = args.risk_mode
    if args.no_risk_overlay:
        profile["oos"]["risk_overlay"] = False
    common = profiles["common"]
    panel = pd.read_csv(PANEL, parse_dates=["trade_date"], dtype={"index_code": str})
    contract = pd.read_csv(CONTRACT)
    target_panel = panel.loc[panel["index_code"].eq(args.index)].copy()
    factor_frame, state_lags = screen.build_factor_frame(target_panel, contract, profiles, STATE_QUALITY)
    registry = screen.candidate_registry(contract, factor_frame, profile, common)
    fields_by_horizon = {h: [item["field"] for item in registry if h in item["allowed_horizons"]] for h in profile["oos"]["horizons_days"]}
    group = target_panel.sort_values("trade_date").set_index("trade_date")
    outputs = []
    fold_outputs = []
    first_valid = group.index[group["close"].notna()].min()
    development_start = group.index[group.index.get_loc(first_valid) + int(profile["oos"]["train_days"]) + max(profile["oos"]["horizons_days"])] if first_valid is not pd.NaT else None
    development_end = pd.Timestamp(profile["oos"]["start"]) - pd.Timedelta(days=1)
    for horizon in profile["oos"]["horizons_days"]:
        for period, start, end in [("training", development_start, development_end), ("backtest", profile["oos"]["start"], group.index.max())]:
            if start is None or pd.Timestamp(start) > pd.Timestamp(end):
                continue
            signal, folds = run_horizon(group, factor_frame, fields_by_horizon[horizon], horizon, profile, start, end, period)
            if signal.empty:
                continue
            signal["benchmark_return"] = group["next_open_to_open"].reindex(signal.index)
            signal["turnover"] = signal["target_position"].diff().abs().fillna(signal["target_position"].abs())
            signal["commission"] = signal["turnover"] * COMMISSION_ONE_WAY
            signal["slippage"] = signal["turnover"] * SLIPPAGE_ONE_WAY
            signal["total_cost"] = signal["commission"] + signal["slippage"]
            signal["gross_strategy_return"] = signal["target_position"] * signal["benchmark_return"]
            signal["strategy_return"] = signal["gross_strategy_return"] - signal["total_cost"]
            signal["index_code"] = args.index
            outputs.append(signal.reset_index())
            fold_outputs.append(folds)
    output_dir = args.output_root / args.index
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    folds = pd.concat(fold_outputs, ignore_index=True) if fold_outputs else pd.DataFrame()
    ledger.loc[ledger["period"].eq("training")].to_csv(output_dir / "training_signal.csv", index=False, encoding="utf-8-sig")
    ledger.loc[ledger["period"].eq("backtest")].to_csv(output_dir / "oos_signal.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output_dir / "fold_audit.csv", index=False, encoding="utf-8-sig")
    summary = {"index": args.index, "index_name": profile["name"], "oos_config": profile["oos"], "training_interval": {"start": str(development_start), "end": str(development_end), "meaning": "rolling training/development evaluation, not in-sample fit returns"}, "backtest_interval": {"start": profile["oos"]["start"], "end": str(group.index.max()), "meaning": "all rolling backtest-fold signals concatenated chronologically before performance evaluation"}, "horizons": {str(h): {period: {"interval": interval_info(ledger.loc[ledger["horizon_days"].eq(h) & ledger["period"].eq(period)]), "metrics": metrics(ledger.loc[ledger["horizon_days"].eq(h) & ledger["period"].eq(period)])} for period in ("training", "backtest")} for h in profile["oos"]["horizons_days"]} if not ledger.empty else {}, "state_lags": state_lags, "selection": "top-k selected inside each training fold only", "commission_one_way": COMMISSION_ONE_WAY, "slippage_one_way": SLIPPAGE_ONE_WAY, "total_cost_one_way": ONE_WAY_COST}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

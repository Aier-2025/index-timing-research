"""Generate the consolidated index timing factor research brief."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"F:\data\index_timing_raw")
FACTOR_ROOT = Path(r"F:\量化因子库")
OUT = Path(r"F:\简报\指数择时因子研究报告_20260826.html")
CONTRACT = FACTOR_ROOT / "docs" / "reports" / "csi1000_factor_term_audit_20260819" / "feature_contract_draft.csv"
REPORT_20260820 = FACTOR_ROOT / "中证1000择时策略回测与利差慢专家探索报告_20260820.html"

INDEX_NAMES = {
    "000905": "中证500",
    "000852": "中证1000",
    "399006": "创业板指",
    "000688": "科创50",
    "932000": "中证2000",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def pct(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):+.{digits}%}"
    except (TypeError, ValueError):
        return "—"


def num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    if frame.empty:
        return "<p class='muted'>暂无可用数据。</p>"
    formats = formats or {}
    rows = []
    headers = "".join(f"<th>{esc(c)}</th>" for c in columns)
    for _, row in frame.loc[:, columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            formatter = formats.get(column)
            if formatter == "pct":
                value = pct(value)
            elif formatter == "num":
                value = num(value)
            elif formatter == "int":
                try:
                    value = f"{int(float(value)):,}"
                except (TypeError, ValueError):
                    value = "—"
            cells.append(f"<td>{esc(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<div class='scroll'><table><thead><tr>" + headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def load_current_oos() -> pd.DataFrame:
    rows = []
    for code, name in INDEX_NAMES.items():
        path = DATA / "index_oos_workers_v2" / code / "summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        for horizon, periods in summary.get("horizons", {}).items():
            values = periods.get("backtest", {}).get("metrics", {})
            interval = periods.get("backtest", {}).get("interval", {})
            rows.append({
                "指数": name,
                "代码": code,
                "期限": int(horizon),
                "区间": f"{interval.get('signal_start', '—')} 至 {interval.get('signal_end', '—')}",
                "样本数": values.get("rows"),
                "年化超额": values.get("annualized_excess"),
                "超额夏普": values.get("active_sharpe"),
                "策略年化": values.get("annualized_return"),
                "基准年化": values.get("benchmark_annualized_return"),
                "最大回撤": values.get("max_drawdown"),
                "年化换手": values.get("annual_turnover"),
            })
    return pd.DataFrame(rows)


def load_candidate_summaries() -> pd.DataFrame:
    paths = [
        ("中证500", "风险预算搜索", DATA / "index_risk_budget_search" / "000905" / "summary.json"),
        ("中证1000", "Ridge模型网格", DATA / "index_model_grid_search" / "000852" / "summary.json"),
        ("创业板指", "风险预算搜索", DATA / "index_risk_budget_search" / "399006" / "summary.json"),
        ("科创50", "Ridge模型网格", DATA / "index_model_grid_search" / "000688" / "summary.json"),
        ("中证2000", "定向规则搜索", DATA / "index_targeted_strategy_search" / "932000" / "summary.json"),
    ]
    rows = []
    for name, method, path in paths:
        if not path.exists():
            continue
        item = read_json(path).get("best", {})
        rows.append({
            "指数": name,
            "候选方法": method,
            "期限": item.get("horizon_days", item.get("holding_days")),
            "方向/策略": item.get("direction", item.get("strategy")),
            "开发期超额": item.get("training_annualized_excess"),
            "开发期超额夏普": item.get("training_active_sharpe"),
            "验证/重放超额": item.get("backtest_annualized_excess", item.get("validation_annualized_excess")),
            "验证/重放超额夏普": item.get("backtest_active_sharpe", item.get("validation_active_sharpe")),
            "状态": "通过双区间筛选" if item.get("passes_gate") else "未通过或仅候选",
        })
    return pd.DataFrame(rows)


def load_frequency_ablation() -> pd.DataFrame:
    rows = []
    for code, name in INDEX_NAMES.items():
        path = DATA / "index_frequency_ablation_20260826" / code / "ablation_results.csv"
        if not path.exists() or code == "932000":
            continue
        data = pd.read_csv(path)
        data = data.loc[data["direction"].eq("direct")].copy()
        if data.empty:
            continue
        for _, row in data.sort_values("replay_score", ascending=False).head(4).iterrows():
            rows.append({
                "指数": name,
                "输入结构": row["mode_label"],
                "方向": row["direction"],
                "开发期超额": row["development_annualized_excess"],
                "开发期超额夏普": row["development_active_sharpe"],
                "历史重放超额": row["replay_annualized_excess"],
                "历史重放超额夏普": row["replay_active_sharpe"],
                "历史重放换手": row["replay_annual_turnover"],
            })
    return pd.DataFrame(rows)


def source_label(source: str) -> str:
    return {
        "macro_csv_2015_plus": "股票宏观因子库",
        "stock_pool_2000_derived": "市场情绪因子/股票池聚合派生",
        "local_index_cache": "宽基指数日线/跨指数派生",
        "public_macro_2005_plus": "公开宏观数据",
    }.get(source, source)


def factor_sections() -> tuple[str, str, str]:
    if not CONTRACT.exists():
        return "<p>因子合同文件不存在。</p>", "", ""
    contract = pd.read_csv(CONTRACT)
    fast = contract.loc[contract["horizon_bucket"].eq("responsive_daily")].copy()
    fast["来源"] = fast["source"].map(source_label)
    source_summary = fast.groupby("来源").size().rename("因子数").reset_index()
    family_summary = fast.groupby("family").size().rename("因子数").reset_index().rename(columns={"family": "因子类别"})
    summary_html = "<div class='grid'>" + "<section class='card'><h3>按来源</h3>" + table(source_summary, ["来源", "因子数"], {"因子数": "int"}) + "</section>" + "<section class='card'><h3>按经济类别</h3>" + table(family_summary, ["因子类别", "因子数"], {"因子数": "int"}) + "</section></div>"
    catalog = fast.rename(columns={
        "field": "字段标识",
        "source": "原始来源",
        "family": "经济类别",
        "availability_lag_days": "可得性滞后",
        "observed_frequency": "观测频率",
        "horizon_bucket": "期限组",
    })
    catalog["原始来源"] = catalog["原始来源"].map(source_label)
    catalog_html = table(catalog, ["字段标识", "原始来源", "经济类别", "可得性滞后", "观测频率", "期限组"], {"可得性滞后": "int"})
    return summary_html, f"<details><summary>展开查看 55 个响应型因子目录</summary>{catalog_html}</details>", f"<p>55 个响应型因子来自合同中 `responsive_daily` 组；该组按观测频率、字段持久性、可得性滞后和经济家族定义，不使用全期绩效反向筛选。Alpha101 不在该 55 因子基线中。</p>"


def main() -> None:
    oos = load_current_oos()
    candidates = load_candidate_summaries()
    ablation = load_frequency_ablation()
    source_html, catalog_html, factor_note = factor_sections()

    fastslow = pd.read_csv(FACTOR_ROOT / "csi1000_fastslow_multihorizon_development_20260820" / "development_comparison.csv")
    fastslow = fastslow.rename(columns={
        "variant": "输入结构", "horizon": "期限", "features": "字段数",
        "dev_annualized_excess": "开发期超额", "dev_active_sharpe": "开发期超额夏普",
    })
    slow_multi = pd.read_csv(FACTOR_ROOT / "csi1000_slow_multiscale_development_20260820" / "development_comparison.csv")
    slow_multi = slow_multi.rename(columns={
        "variant": "输入结构", "horizon": "期限", "features": "字段数",
        "annualized_excess": "开发期超额", "active_sharpe": "开发期超额夏普",
        "annual_turnover": "开发期换手",
    })
    six_summary = read_json(FACTOR_ROOT / "csi1000_six_expert_execution_oos_matured_20260820" / "summary.json")

    current_table = table(
        oos.sort_values(["代码", "期限"]),
        ["指数", "期限", "区间", "样本数", "年化超额", "超额夏普", "策略年化", "基准年化", "最大回撤", "年化换手"],
        {"样本数": "int", "年化超额": "pct", "超额夏普": "num", "策略年化": "pct", "基准年化": "pct", "最大回撤": "pct", "年化换手": "num"},
    )
    candidate_table = table(
        candidates,
        ["指数", "候选方法", "期限", "方向/策略", "开发期超额", "开发期超额夏普", "验证/重放超额", "验证/重放超额夏普", "状态"],
        {"开发期超额": "pct", "开发期超额夏普": "num", "验证/重放超额": "pct", "验证/重放超额夏普": "num"},
    )
    ablation_table = table(
        ablation,
        ["指数", "输入结构", "开发期超额", "开发期超额夏普", "历史重放超额", "历史重放超额夏普", "历史重放换手"],
        {"开发期超额": "pct", "开发期超额夏普": "num", "历史重放超额": "pct", "历史重放超额夏普": "num", "历史重放换手": "num"},
    )
    fastslow_table = table(
        fastslow,
        ["输入结构", "期限", "字段数", "开发期超额", "开发期超额夏普"],
        {"开发期超额": "pct", "开发期超额夏普": "num", "字段数": "int"},
    )
    slow_table = table(
        slow_multi,
        ["输入结构", "期限", "字段数", "开发期超额", "开发期超额夏普", "开发期换手"],
        {"开发期超额": "pct", "开发期超额夏普": "num", "字段数": "int", "开发期换手": "num"},
    )

    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>指数择时因子研究报告 2026-08-26</title>
<style>
:root{{--ink:#17212b;--muted:#5b6870;--line:#d5dde3;--teal:#123b45;--wash:#eef6f6;--good:#087f5b;--bad:#b42318;--warn:#b45309}}
*{{box-sizing:border-box}} body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1600px;margin:0 auto;padding:0 22px 140px;color:var(--ink);line-height:1.75;background:#fff}}
h1,h2,h3{{color:var(--teal)}} h1{{margin:0 0 8px}} h2{{border-bottom:2px solid #8dc4c2;padding-bottom:6px;margin-top:42px}}
.nav{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:10px 0;display:flex;gap:18px;overflow:auto;white-space:nowrap}} .nav a{{color:#245563;text-decoration:none;font-size:14px}}
.hero{{padding:30px 0 24px;border-bottom:1px solid var(--line)}} .lead{{font-size:18px;color:#30414d;max-width:1120px}}
.note{{background:#fff7ed;border-left:4px solid var(--warn);padding:14px 18px}} .rule{{background:var(--wash);border-left:4px solid #2878a8;padding:12px 16px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} .card{{border:1px solid var(--line);padding:16px;border-radius:6px;background:#fbfcfd}} .card h3{{margin-top:0}}
.scroll{{max-width:100%;overflow:auto}} table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}} th,td{{border:1px solid var(--line);padding:8px;text-align:right;vertical-align:top}} th:first-child,td:first-child{{text-align:left}} th{{background:#e8f2f4}}
tr:first-child td{{background:#f0faf5}} .muted{{color:var(--muted)}} .good{{color:var(--good);font-weight:800}} .bad{{color:var(--bad);font-weight:800}} code{{overflow-wrap:anywhere}}
details{{border:1px solid var(--line);border-radius:6px;padding:0 14px;margin:14px 0}} summary{{cursor:pointer;font-weight:700;padding:12px 0}}
@media(max-width:760px){{body{{padding:0 14px 100px}}.grid{{grid-template-columns:1fr}}table{{font-size:12px}}}}
</style></head><body>
<nav class="nav"><a href="#结论">结论</a><a href="#范围">研究范围</a><a href="#流程">研究流程</a><a href="#指数结果">指数结果</a><a href="#55因子">55因子</a><a href="#慢因子">慢因子</a><a href="#消融">频率消融</a><a href="#局限">局限与计划</a></nav>
<header class="hero" id="结论"><h1>指数择时因子研究报告</h1>
<p class="lead">本报告整合五指数日频择时项目截至 2026 年 8 月 26 日的研究成果，并吸收《中证1000择时策略回测与利差慢专家探索报告（2026-08-20）》的因子期限、55 快因子和慢因子探索结论。</p>
<div class="note"><b>结论先行：</b>当前最有持续证据的是中证1000 10日滚动模型和创业板指 10日滚动模型；中证500当前滚动全因子模型未形成正的主动超额，科创50短历史下不稳定，中证2000样本过短且完整 OOS 未通过。中证1000 的 55 个响应型快因子模型在原 ETF 执行研究中曾取得冻结区间年化超额 +6.35%、超额夏普 +0.17；慢因子与利差分支改善回撤但未提升成本后主动收益，因此暂不接入主模型。</div></header>

<section id="范围"><h2>1. 研究范围与数据口径</h2>
<div class="grid"><section class="card"><h3>研究对象</h3><ul><li>中证500：有效行情起点 2007-01-15。</li><li>中证1000：有效行情起点 2014-10-17。</li><li>创业板指：有效行情起点 2010-06-01。</li><li>科创50：有效行情起点 2020-07-23。</li><li>中证2000：有效行情起点 2023-08-11。</li></ul></section>
<section class="card"><h3>执行与成本</h3><ul><li>指数使用原始价格点位，不做前复权、不计分红。</li><li>信号在 T 日收盘后形成，默认 T+1 日执行。</li><li>成本代理为单边手续费 3 bp、单边滑点 5 bp，合计单边 8 bp。</li><li>滚动训练中的标签成熟、缺失填补、截尾、标准化和因子选择均只使用历史训练窗口。</li></ul></section></div>
<div class="rule">报告中的 OOS 结果均指已经运行过的滚动回测或历史重放。由于这些日期已被研究观察，后续改动不能把它们重新表述为全新的样本外证据。</div></section>

<section id="流程"><h2>2. 完整研究流程</h2>
<ol><li><b>数据审计：</b>统一五指数字段、有效日期、数据源和缺失结构，保留原始数据 manifest。</li><li><b>时点合同：</b>将因子按可得性滞后、观测频率、持久性和经济家族分为响应型、 中频、持久/滞后持久和慢发布。</li><li><b>因子筛选：</b>指数独立筛选候选因子；滚动 worker 在每个训练折内重新选 Top-K，避免全样本筛选泄露。</li><li><b>策略搜索：</b>比较 Ridge 五档仓位、风险预算规则、定向趋势/反转规则及多种期限。</li><li><b>滚动 OOS：</b>各指数独立设置训练窗、期限和 OOS 起点，按完整拼接区间评价年化超额、主动夏普、回撤和成本。</li><li><b>中证1000 专项：</b>进一步研究 55 快因子、慢金融条件因子、六专家、利差状态和慢规则专家。</li><li><b>当前暂停项：</b>快主模型与慢调整模型的并行/串行结构及 fold 长度网格暂不继续训练，待慢因子统计审计完成后再恢复。</li></ol></section>

<section id="指数结果"><h2>3. 五指数滚动 OOS 结果</h2>
<p>下表来自项目的独立滚动 worker，按完整回测区间拼接后评价；“年化超额”是相对指数买入持有的成本后主动收益，“超额夏普”是主动收益夏普。</p>
{current_table}
<h3>既有候选搜索摘要</h3>{candidate_table}
<p class="muted">当前重点：中证1000 10日、创业板指 10日。中证500和科创50不能仅凭开发期或规则局部结果宣称稳定有效；中证2000从 2025 年起的样本长度和主动指标均不足以支持复杂模型扩展。</p></section>

<section id="55因子"><h2>4. 中证1000：55 个响应型快因子基线</h2>
{factor_note}
<p>55 快因子模型使用 LightGBM 预测未来可执行收益，并映射为五档仓位。原报告开发期比较中，10日快因子模型年化超额 +11.54%、超额夏普 +0.78；冻结区间 2024-01-02 至 2026-07-15 年化超额 +6.35%、超额夏普 +0.17，基准最大回撤约 -26.67%，策略最大回撤约 -18.67%。</p>
<div class="grid"><section class="card"><h3>来源归类</h3>{source_html}</section><section class="card"><h3>适用边界</h3><p>这 55 个字段不来自 Alpha101。Alpha101 更适合股票横截面技术因子；当前 55 快因子是市场级时间序列输入，覆盖市场广度、成交/流动性、风险偏好、量价和波动。若后续迁移 Alpha101，必须先定义股票池聚合和时点口径。</p></section></div>
{catalog_html}</section>

<section id="慢因子"><h2>5. 快慢期限与慢专家探索</h2>
<p>2026-08-20 的中证1000专项研究显示，55 快因子在 10日期限的开发期表现优于直接加入全慢因子；慢因子更适合描述金融条件、资金价格和风险状态，不适合未经约束地与快因子平权拼接。</p>
{fastslow_table}
<h3>慢变量多尺度比较</h3>{slow_table}
<h3>慢专家和利差分支 OOS</h3>
<div class="grid"><section class="card"><h3>六专家 + 利差状态层</h3><p>2024-01-02 至 2026-07-15：年化超额 {pct(six_summary.get('annualized_excess_return'))}，超额夏普 {num(six_summary.get('active_sharpe'))}，最大回撤 {pct(six_summary.get('etf_max_drawdown'))}；相对基准回撤改善，但主动收益和主动夏普为负。</p></section>
<section class="card"><h3>研究判断</h3><p>慢因子可能具备防御/状态识别价值，但当前证据不支持把它作为方向预测器或直接替代快模型。慢因子统计画像和低复杂度调整模型已记录为后续任务，尚未形成正式结论。</p></section></div></section>

<section id="消融"><h2>6. 频率分组消融：历史重放诊断</h2>
<p>本项目新增的频率消融比较了技术因子、55快因子、快+中、快+慢原始、快+慢变换、全量和快慢专家融合。由于使用的是已观察过的项目历史区间，以下只作为方法诊断和归因，不作为新的 OOS 选参依据。</p>
{ablation_table}
<div class="rule">总体规律：直接把大量慢变量拼进单一模型会抬高部分开发期指标，但在历史重放中普遍回落；中证1000 的快慢双专家融合相对更接近正向，但证据仍不足以替换既有基线。下一步应先验证慢因子是否存在稳定统计预测能力，再决定是否使用极简状态层。</div></section>

<section id="局限"><h2>7. 局限、已暂停工作与后续计划</h2>
<ul><li><b>短历史：</b>科创50和中证2000的样本长度明显短于中证500/创业板指，开发期高夏普不能与长历史证据等同。</li><li><b>指数执行代理：</b>中证1000专项研究使用 512100 ETF 执行代理；指数点位研究与可交易 ETF 执行存在基差、流动性和交易时段差异。</li><li><b>慢因子复杂度：</b>慢变量数量大、持久性强、内部冗余高，复杂快慢融合容易过拟合。快主模型 + 慢独立调整模型、不同期限组合和不同 fold 长度暂时记录，待统计审计完成后再继续。</li><li><b>Alpha101 边界：</b>当前主结果没有使用 Alpha101；若后续使用，必须先完成横截面到指数时间序列的聚合定义和独立验证。</li><li><b>实盘前要求：</b>建立冻结配置、未观察数据 OOS、执行审计、成本敏感性和独立复核，不能直接把本报告候选视为实盘模型。</li></ul>
<p class="muted">主要来源：项目 `docs/five_index_timing_strategy_report_20260825.html`、`docs/index_agent_oos_report_20260824.md`、`F:\\量化因子库\\中证1000择时策略回测与利差慢专家探索报告_20260820.html` 及其配套实验目录。</p></section>
</body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

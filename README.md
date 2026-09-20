# 五指数日频择时研究

研究对象为中证500、中证1000、创业板指、科创50和中证2000。项目使用原始指数价格点位进行日频研究，不做前复权、不计分红；原始行情保存在本地 `F:/data/index_timing_raw/`，不提交到 GitHub。

## 当前数据

- JoinQuant：中证500、创业板指、中证1000、科创50日频原始行情。
- TonglianData：中证2000日频原始行情。
- 真实有效起点、字段缺失和下载来源见 `F:/data/index_timing_raw/index_timing_data_audit.json`。

## 研究约束

- 时间轴为五个指数有效交易日的并集。
- 2015 年之前不使用宏观、情绪和资金类因子。
- 所有字段按可得时点滞后，信号在下一交易日开盘执行。
- 慢专家使用 20/60 日及六期限实验，短历史指数单独评估统计功效。

## 使用

```powershell
python scripts/audit_index_timing_data.py
python scripts/build_daily_research_panel.py
pytest -q
```

详细设计见 `docs/research_design.md`。

训练集与回测集纪律见 `docs/research_lifecycle_protocol.md`。所有回测结果均按完整拼接区间整体评价。

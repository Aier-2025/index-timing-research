# Handoff

日期：2026-08-18

## 当前目标

为中证500、中证1000、科创50、创业板指和中证2000构建无未来函数的日频择时研究链；分钟级数据暂不作为探索阶段依赖，分红暂不计入。

## 已完成

- 已整理指数背景、基日、发布信息、数据源和口径，见 `research_notes.md`。
- 已确认指数点位没有原生前复权字段；回测含分红应使用全收益指数日线，分钟价格指数用于构建盘中信号。
- 已在聚宽 research 环境实测：指数证券表中可见中证500、中证1000、科创50、创业板指价格指数；未见中证2000和五个目标的全收益指数条目。
- 已实现远端 JSONL 阶段日志及本地轮询同步原型，日志链路能记录 `batch_started` 和 `get_price_started`。
- 2026-08-24：重新通过 JoinQuant 下载四个可用指数的日频原始数据，并通过 TonglianData `mkt_idxd_csi` 补齐中证2000。
- 2026-08-24：完成五指数并集研究面板、2015 年前宏观/情绪因子禁用规则和未来函数验证。

## 已验证失败的路径

- 聚宽 `get_price(..., frequency='1m')` 下载上述指数分钟历史：全历史、单年和单月请求均长期停在 `get_price_started`，没有返回行数或数据产物。
- 已停止所有相关聚宽下载与日志同步进程；未产生有效行情 ZIP、CSV 或 manifest。
- AkShare/东方财富不作为全量历史分钟主源：其分钟历史覆盖有限，且本机沙箱网络请求被阻断。

## 当前文件

- `scripts/download_four_indices_joinquant.py`：旧全量尝试脚本，勿直接运行。
- `scripts/download_index_minute_batch_joinquant.py`：月度分钟批处理和远端 JSONL 日志原型。
- `scripts/jq_remote_progress_monitor.py`：本地远端日志镜像原型。
- `scripts/index_minute_batch.json`：最后一次探针配置（中证500，2005-01）。
- `F:\data\four_index_timing_selection\joinquant_indices\`：仅包含 runner/验证日志，无有效行情产物。

## 下一步

1. 在五指数日频面板上按指数独立运行技术因子、市场状态因子和慢专家的滚动样本外研究。
2. 对 2015 年后的宏观/情绪字段逐字段应用 `availability_lag_days`，阻止未完成时点审计的异常因子进入模型。
3. 以固定开发期和冻结 OOS 协议比较 1/3/5/10/20/60 日专家，不把短历史指数的结果与长历史指数直接混合。

## 约束

- 当前无可用下载结果，不得把任何 `minute_batches` 日志当作行情数据。
- 不对指数点位调用或伪造 `fq='pre'`；只有使用价格指数与全收益指数构造的日级因子，才可生成合成前复权展示序列。

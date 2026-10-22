# 研究四：EvolveTrade 云端复现项目

本项目是研究四正式实验的独立云端版本，使用真实市场价格、真实新闻数据和 DeepSeek API，不使用合成数据或模拟代理。项目包含当前 `result/runs` 实验断点，可迁移到云服务器后继续运行。

## 项目内容

```text
coding/                  实验运行、敏感性分析、长期实验、表格和图生成脚本
config/                  实验配置
data/real/               真实价格、OHLCV、高频月度数据和新闻数据
result/runs/             当前正式矩阵断点与已完成结果
src/evolvetrade/         LLM 代理、工具调用、回测和指标实现
requirements.txt         Python 依赖
```

主矩阵配置为 6 个市场窗口、5 种方法、3 个 seed，共 90 个实验单元：

```text
2025-01, 2025-04, 2025-09, 2025-11, 2026-02, 2026-04
Static Base, Static Tool-Calling, EvolveBase, EvolveStrategy, EvolveTrade
seeds: 101, 202, 303
```

## 云服务器准备

建议使用 Python 3.11 或 3.12、4 个以上 CPU、16 GB 以上内存和持久化磁盘。项目目录必须整体保留，尤其是 `result/runs` 和 `data/real`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY='你的DeepSeek API Key'
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY='你的DeepSeek API Key'
```

Linux云服务器使用相对路径运行，项目不依赖Windows盘符：

```bash
cd /path/to/研究四_云端复现项目
export PYTHONUTF8=1
export DEEPSEEK_API_KEY='你的DeepSeek API Key'
python -m compileall -q src coding
```

数据目录、结果目录和断点目录均由项目根目录自动解析，不要修改代码中的路径。云端必须使用持久化磁盘保存 `data/real/` 和 `result/`。建议使用 `tmux` 或 `screen`，避免SSH断开导致任务终止。

## 从断点继续主实验

在项目根目录运行：

```bash
python coding/5_run_llm_experiments.py
```

脚本会读取各实验目录下的 `records.jsonl`，已经完成的 `decision_date` 会自动跳过。工具调用过程还会写入对应目录的 `tool_checkpoint_YYYY-MM-DD.json`，中断后可恢复当前决策过程。不要同时启动两个主矩阵进程，否则会造成同一结果文件竞争写入。

后台运行示例：

```bash
nohup python coding/5_run_llm_experiments.py > result/formal_matrix.log 2>&1 &
```

推荐使用 `tmux` 或 `screen`，并将 `result/` 所在磁盘设置为持久化磁盘。

## 主矩阵完成后的顺序

主矩阵完成后按以下顺序运行：

```bash
python coding/6_run_interval_sensitivity.py
python coding/7_run_long_horizon.py
python coding/8_generate_paper_tables.py
python coding/9_generate_paper_figures.py
python coding/11_run_a_share_validation.py
python coding/13_generate_a_share_results.py
```

这些步骤生成更新间隔敏感性、长期实验、论文表 1--7、图 2--7，以及 A 股验证结果。只有主矩阵完整后才运行后续脚本。

## 全流程一键运行

完整流程按阶段运行并显示阶段进度；任何阶段失败都会停止，不会生成伪造的后续结果：

```bash
python coding/16_run_full_pipeline.py
```

从断点继续后处理：

```bash
python coding/16_run_full_pipeline.py --from-stage interval_sensitivity
```

主矩阵本身仍然使用 `result/runs` 内的逐日记录和工具检查点恢复。若主矩阵已经部分完成，重新运行 `5_run_llm_experiments.py` 会显示当前单元进度并跳过已经落盘的决策日。

## 创新完整冒烟链

冒烟测试不是简化的几步试跑，而是在短真实样本上执行完整流程，用于验证所有功能和结果输出是否可运行。它不替代正式长样本实验，也不改写原始基线代码。

```bash
python coding/15_innovation_full_smoke.py
```

该流程读取 `data/real/us_daily_adjusted_close.csv`，执行数据审计、统一交易成本、权重约束、固定基线、情景识别、失效触发、经验门控、滚动样本外分段、回撤回滚接口、消融比较、配对Bootstrap和结果可视化。输出位于 `result/innovation_full_smoke/`：

```text
tables/table_1_metrics.csv
tables/table_2_event_trace.csv
tables/table_3_bootstrap.json
tables/table_4_ablation.csv
tables/table_5_rolling_oos.csv
tables/table_6_data_audit.csv
tables/table_7_significance.csv
figures/figure_1_equity_and_difference.png
figures/figure_2_failure_triggers.png
```

冒烟链中的 `FixedLeverage` 是必要的控制组，用于识别创新是否只是杠杆效应；正式论文只能在多窗口、多seed和样本外结果完成后报告创新结论。`5_run_deepseek_smoke.py`、`3_make_report.py` 等旧演示入口不属于正式结果链，保留用于历史复现，不得混入正式统计。

## 结果检查

每个方法、窗口和 seed 的目录结构为：

```text
result/runs/<window>/<method>/protocol_v2/<seed>/
    records.jsonl
    metrics.csv
    tool_checkpoint_*.json
```

`records.jsonl` 保存逐日决策、组合权重、工具调用、资产收益和策略更新；`metrics.csv` 保存 Sharpe、累计收益、最大回撤、胜率、波动率、交易成本后指标、换手率和平均现金比例。

快速查看最新断点：

```bash
find result/runs -name records.jsonl -printf '%T@ %p\\n' | sort -nr | head
```

PowerShell：

```powershell
Get-ChildItem result/runs -Recurse -Filter records.jsonl |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName,LastWriteTime,Length
```

## 数据与复现边界

价格数据位于 `data/real/us_daily_adjusted_close.csv`、`us_daily_ohlcv.parquet` 和 `hf_monthly/`；新闻数据位于 `data/real/news/`。实验使用决策日以前可获得的信息，交易成本由配置中的 `transaction_cost_bps` 控制。云端运行时应固定项目版本、Python 版本、依赖版本和 API 模型，并保留完整日志与 `result/runs` 目录。

## 迁移断点

从本项目复制到云服务器时，至少保留：

```text
coding/5_run_llm_experiments.py
config/base_config.json
data/real/
src/
result/runs/
requirements.txt
```

复制完成后先运行一次语法检查，再启动主矩阵：

```bash
python -m compileall src coding
python coding/5_run_llm_experiments.py
```

# Wind Power Forecasting

该仓库用于组织风电功率预测项目的本地开发、GitHub 同步、超算作业投递与结果回传分析流程。

## 当前目录约定

- `src/wind_forecast/`: 通用 Python 源码，放可复用的数据处理、建模、训练、推理与工具函数。
- `scripts/`: 可直接运行的脚本入口，按 `data`、`train`、`eval`、`sync` 分类。
- `configs/`: 公共配置与不同预测时距的配置模板。
- `jobs/`: 超算作业模板与 `slurm` 提交脚本。
- `tests/`: 本地测试代码。
- `docs/reports/worklogs/`: 每次工作完成后的 Markdown 报告。
- `data/`: 数据清单、样例与说明，不存放原始敏感数据。
- `transfer/to_hpc/payload/`: 需要上传到服务器的数据或文件暂存区。
- `transfer/from_hpc/`: 从服务器拉回的结果、日志或摘要暂存区。
- `artifacts/`: 本地调试产物和从服务器拉回的模型/结果归档。
- `15min-lead/`, `4h-lead/`, `24h-lead/`: 按预测时距划分的实验工作区，保留现有 `power/` 和 `wind/` 目录。

## 工作方式

- 统一规则见 [AGENTS.md](AGENTS.md)。
- 本地可用数据根目录为 `C:\Users\caosh\Desktop\WTPC\data`。
- 原始数据、私有结果和大体积模型文件不进入 Git。
- 涉及深度学习训练的代码先本地编写和最小化测试，再同步 GitHub，随后由服务器拉取执行。


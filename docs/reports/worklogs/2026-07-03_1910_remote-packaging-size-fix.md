# 本次目标

- 修复服务器端 `package_remote_run.py` 打包后无法推送到 GitHub 的问题。

# 实际修改了什么

- 修改 `15min-lead/wind/src/xinyang_wind15/remote_results.py`
  - 默认不再复制 `val_predictions.csv` 与 `test_predictions.csv` 到仓库结果目录。
  - 新增 `include_predictions` 开关，确有需要时才显式包含完整预测明细。
  - 在 `manifest.json` 的 `excluded_large_files` 中显式记录被排除的预测文件。
- 修改 `15min-lead/wind/scripts/package_remote_run.py`
  - 新增 `--include-predictions` 参数。
  - 默认行为改为“仓库安全摘要模式”。
- 修改 `15min-lead/wind/tests/test_remote_results.py`
  - 调整默认测试，验证预测 CSV 默认不入库。
  - 新增测试，验证显式开启 `include_predictions` 时仍可复制预测明细。

# 为什么这样做

- 根因是打包脚本把完整的逐时刻预测 CSV 一起放进了仓库目录。
- 淮安 `1min` 多机组联合预测的 `val_predictions.csv` / `test_predictions.csv` 已超过 GitHub 的 `100 MB` 单文件限制，因此 push 必然失败。
- 这些明细文件适合保留在服务器产物目录或单独压缩归档，不适合直接进入 Git 仓库。仓库里应优先保留可版本化的轻量摘要。

# 做了哪些验证

- 计划执行：
  - `python -m pytest 15min-lead/wind/tests/test_remote_results.py`
  - `python 15min-lead/wind/scripts/package_remote_run.py -h`
- 这次修复不改训练逻辑，只改结果打包策略。

# 当前风险、阻塞和下一步建议

- 服务器端此前已经生成的超大预测 CSV 仍保留在 `artifacts/server_runs/...` 中，但这没有问题；只要重新执行新的打包脚本，仓库内不会再复制它们。
- 如果服务器端已经提交了包含大文件的本地 commit，需要先把该 commit 从本地历史里拿掉，再重新打包和提交。
- 建议后续默认只 push：
  - `metrics.*`
  - `training_history.csv`
  - `val_per_turbine.csv`
  - `test_per_turbine.csv`
  - `summary.json`
  - `store_summary.json`
  - 裁剪后的日志和 `manifest.json`

# 涉及的数据、服务器或 GitHub 操作

- 本次不涉及原始数据修改。
- 需要补一次 Git 提交和推送，供服务器端 `git pull` 后重新打包。

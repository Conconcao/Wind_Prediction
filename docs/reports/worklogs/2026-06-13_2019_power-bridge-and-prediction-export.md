# 本次目标

- 复用 `WTPC` 项目的多变量功率曲线思路，在仅使用基础特征的前提下，把风速预测映射成功率预测。
- 为当前风速预测训练脚本补齐逐样本 `val/test` 预测导出能力，打通后续“服务器训练 -> 仓库回传 -> 本地功率评估”的统一链路。

# 实际修改了什么

- 在 `15min-lead/wind/src/xinyang_wind15/sequence.py` 中扩展了 `evaluate_window_model`：
  - 支持在评估阶段按需返回逐样本预测表。
  - 新增 `build_window_prediction_frame(...)`，统一生成 `origin_timestamp / target_timestamp / turbine_id / split / y_true / y_pred` 格式。
- 更新以下训练脚本，在最终评估时额外写出 `val_predictions.csv` 和 `test_predictions.csv`：
  - `15min-lead/wind/scripts/train_gwnet_from_store.py`
  - `15min-lead/wind/scripts/train_agcrn_from_store.py`
  - `15min-lead/wind/scripts/train_mtgnn_from_store.py`
  - `15min-lead/wind/scripts/train_moderntcn_from_store.py`
  - `15min-lead/wind/scripts/train_tcn_from_store.py`
  - `15min-lead/wind/scripts/train_gru_baseline.py`
  - `15min-lead/wind/scripts/train_tcn_baseline.py`
- 更新 `15min-lead/wind/src/xinyang_wind15/remote_results.py`，让远端打包默认把 `val_predictions.csv / test_predictions.csv` 一并带回仓库。
- 新增 `15min-lead/wind/src/xinyang_wind15/power_curve_bridge.py`：
  - 复用 `WTPC` 的 `build_model`、时间循环特征、工况分段和功率侧指标定义。
  - 固定“基础特征版”功率曲线输入：`ws + turbine_id + 机组静态信息 + 时间循环特征 + ws_regime`。
- 新增 `15min-lead/wind/scripts/evaluate_power_from_ws_predictions.py`：
  - 输入统一的风速预测 CSV/Parquet。
  - 自动对接 `WTPC` 的 `exp110_xinyang_split721_direct/prepared` 切分数据。
  - 输出 `power_predictions.csv`、`overall_metrics.csv`、`per_turbine_metrics.csv`、`per_split_metrics.csv`、`curve_sanity_metrics.csv`、`summary.json`。
- 新增 `15min-lead/wind/scripts/export_gwnet_predictions_from_run.py`：
  - 直接从已有 `Graph WaveNet` 训练目录中的 `gwnet_baseline.pt` 和 `summary.json` 回放推理。
  - 补写 `val_predictions.csv / test_predictions.csv`，避免仅为了导出预测而整轮重训。

# 为什么这样做

- 现有 `remote_runs` 目录只有聚合风速指标，没有逐样本 `y_pred`，所以无法直接做功率映射。这不是分析问题，而是产物链路不完整，必须先补导出。
- 功率映射不能依赖风向、偏航、湍流代理等当前风速模型没有同步预测的信息，否则部署时不可用，也会引入信息不一致。
- 直接复用 `WTPC` 的时间特征、工况分段和功率指标定义，可以保证功率侧结果与既有 WTPC 主线可比，不重新发明一套评价口径。

# 做了哪些验证

- 运行 `python -m py_compile` 检查新增和修改脚本，已通过。
- 使用本地现成风速预测结果做了两轮最小验证：
  - `persistence_test_predictions.csv + lgbm basic MVPC`
    - `RMSE = 207.81 kW`
    - `nRMSE = 0.0831`
    - `Q = 94.40`
  - `lightgbm_test_predictions.csv + lgbm basic MVPC`
    - `RMSE = 227.98 kW`
    - `nRMSE = 0.0912`
    - `Q = 93.86`
- 额外检查了基础特征版功率曲线本身的上限表现（使用真实风速）：
  - `val nRMSE = 0.0270`, `Q = 98.23`
  - `test nRMSE = 0.0354`, `Q = 97.47`

# 当前风险、阻塞和下一步建议

- 当前阻塞不在功率曲线脚本，而在历史服务器结果没有导出 `test_predictions.csv`。因此 `6058/6079` 等已经完成的深度模型结果，暂时还不能直接在本地转成功率指标。
- 对于 `GWNet`，这个阻塞已经被新脚本部分解除：只要服务器端训练目录仍保留 checkpoint 和 store，就可以直接补导出预测，不必重训。
- 下一步建议：
  - 对 `6079` 先优先使用 `export_gwnet_predictions_from_run.py` 补导出预测。
  - 只有在训练目录或 checkpoint 已被清理时，才退回到重训方案。
  - 运行 `package_remote_run.py` 重新打包，把 `test_predictions.csv` 提交回仓库。
  - 本地再执行 `evaluate_power_from_ws_predictions.py`，即可得到目标模型的功率侧 `RMSE / nRMSE / 合格率`。

# 涉及的数据、服务器或 GitHub 操作

- 读取了 `WTPC` 本地 prepared 数据：
  - `C:\Users\caosh\Desktop\WTPC\results\exp110_xinyang_split721_direct\prepared`
- 复用了 `WTPC` 本地代码：
  - `wtpc_dynamic.models`
  - `wtpc_dynamic.evaluate`
  - `wtpc_dynamic.xinyang_e12_utils`
- 本次代码已在本地提交并推送到 GitHub 远端仓库，便于服务器端直接 `git pull --rebase origin main` 后补导出 `6079` 预测。
- `AGENTS.md` 未纳入任何暂存或提交范围。

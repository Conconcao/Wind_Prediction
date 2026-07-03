# 本次目标

- 为淮安场站新增“`1min` 风速历史 + `15min` 风向快照”的 overlap-only 时空联合建模管线。
- 重点接通现有 `GWNet` 的 `dynamic_directional_support` 能力。

# 实际修改了什么

- 修改 `15min-lead/wind/src/xinyang_wind15/loading.py`
  - 新增公开接口 `load_scada_15min_direction()`，支持直接读取 `15min` 风向数据并做时间/机组筛选。
- 修改 `15min-lead/wind/src/xinyang_wind15/raw_one_min.py`
  - 新增 `merge_direction_15min_snapshots()`，按机组对分钟级样本做 backward asof 对齐。
  - 扩展 `build_raw_one_min_feature_frame()`：
    - 支持从 `wd_mean / wd_std / nacelle_mean / nacelle_std` 生成方向派生特征；
    - 自动生成 `wd_sin/cos`、`nacelle_sin/cos`、`yaw_error_sin/cos/abs`。
- 修改 `15min-lead/wind/scripts/build_raw_one_min_store.py`
  - 若配置中提供 `scada_15min_direction`，则自动读取并合并方向快照。
  - 在 `summary.json` 中记录方向快照路径与覆盖率。
- 修改 `15min-lead/wind/tests/test_raw_one_min.py`
  - 新增方向快照 backward 对齐测试。
  - 新增方向派生特征生成测试。
- 新增 overlap-only 配置：
  - `15min-lead/wind/configs/splits/huaian_1min_raw_wsdir_overlap_7_1_2.yaml`
  - `15min-lead/wind/configs/splits/huaian_1min_raw_wsdir_overlap_7_1_2_server.yaml`
- 新增服务器脚本：
  - `jobs/lsf/huaian_build_store_1min_raw_wsdir_overlap.lsf`
  - `jobs/lsf/huaian_train_gwnet_1min_raw_wsdir_overlap.lsf`

# 为什么这样做

- 当前淮安 `1min` 原始 SCADA 中没有可靠的分钟级风向，因此不能直接做“分钟级方向图”。
- 但 `15min` 风向文件完整覆盖 `20` 台机组，且方向字段无缺失，足以作为分钟级 origin 的最近快照信息。
- 对当前任务最合适的方案不是改成纯 `15min` 预测，而是：
  - 继续保留 `1min` 风速历史序列；
  - 用最近一条 `15min` 风向/机舱位置快照补充空间流向信息；
  - 在 `GWNet` 中开启动态方向支撑。

# 做了哪些验证

- 单元测试：
  - `python -m pytest 15min-lead/wind/tests/test_raw_one_min.py`
  - 结果：`6 passed`
- 本地方向版 store smoke：
  - `python 15min-lead/wind/scripts/build_raw_one_min_store.py --config 15min-lead/wind/configs/splits/huaian_1min_raw_wsdir_overlap_7_1_2.yaml --max-turbines 2 --tail-timestamps 1200 --output-dir 15min-lead/wind/artifacts/local_debug/huaian_1min_raw_wsdir_overlap_store_smoke`
  - 结果：成功生成 `12` 个特征：
    - `ws`
    - `wd_mean`
    - `wd_std`
    - `wd_sin/cos`
    - `nacelle_mean`
    - `nacelle_std`
    - `nacelle_sin/cos`
    - `yaw_error_sin/cos/abs`
  - `direction_snapshot_coverage = 1.0`
- 本地方向版 `GWNet` smoke：
  - `python 15min-lead/wind/scripts/train_gwnet_from_store.py --store-dir 15min-lead/wind/artifacts/local_debug/huaian_1min_raw_wsdir_overlap_store_smoke --support-mode distance_correlation --dynamic-directional-support --direction-support-source wd_mean --direction-support-sigma-deg 35.0 --epochs 1 --batch-size 64 --residual-channels 16 --dilation-channels 16 --skip-channels 32 --end-channels 64 --kernel-size 2 --blocks 1 --layers 1 --output-dir 15min-lead/wind/artifacts/local_debug/huaian_1min_gwnet_wsdir_overlap_smoke`
  - 结果：训练与评估正常完成，`summary.json` 中已记录：
    - `dynamic_directional_support = true`
    - `direction_support_source = wd_mean`

# 当前风险、阻塞和下一步建议

- 这条方向增强管线是 overlap-only：
  - 起点：`2023-12-02 00:15:00`
  - 终点：`2025-07-03 23:59:00`
- 因此它不能直接和此前“全窗 `ws-only`”结果混作同一主表，需要单独对照。
- 下一步建议：
  - 提交并推送这批代码；
  - 服务器端先跑 `huaian_build_store_1min_raw_wsdir_overlap.lsf`
  - 再跑 `huaian_train_gwnet_1min_raw_wsdir_overlap.lsf`
  - 训练完成后再与当前 `huaian_gwnet_1min_raw_ws_6308` 做公平对照。

# 涉及的数据、服务器或 GitHub 操作

- 使用的本地方向文件：
  - `C:\Users\caosh\Desktop\WTPC\data\huaian\ALL_TURBINES_DIRECTION_15min_20231202-20240703_QC1.parquet`
- 使用的本地元数据：
  - `D:\Power_prediction\Data\风机基本信息汇总.csv`
- 本次尚未执行 Git 提交或推送。

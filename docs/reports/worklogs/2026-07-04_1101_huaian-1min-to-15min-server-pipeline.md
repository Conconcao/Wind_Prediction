# 本次目标

- 为淮安场站新增一条“先将 `1min` 原始观测聚合为 `15min`，再按 `15min-lead` 口径建模”的正式服务器流程。
- 给出可直接提交的 `LSF/bsub` 脚本。

# 实际修改了什么

- 在 `15min-lead/wind/src/xinyang_wind15/loading.py` 新增：
  - `aggregate_scada_1min_to_15min`
  - 圆周角均值与圆周角标准差辅助函数
- 新增聚合脚本：
  - `15min-lead/wind/scripts/aggregate_scada_1min_to_15min.py`
- 新增淮安 `1min -> 15min` 配置：
  - `15min-lead/wind/configs/splits/huaian_15min_from_1min_7_2_1.yaml`
  - `15min-lead/wind/configs/splits/huaian_15min_from_1min_7_2_1_server.yaml`
- 新增服务器作业脚本：
  - `jobs/lsf/huaian_aggregate_1min_to_15min.lsf`
  - `jobs/lsf/huaian_build_store_hubws_from_1minagg.lsf`
  - `jobs/lsf/huaian_train_gwnet_hubws_from_1minagg.lsf`
- 更新测试：
  - `15min-lead/wind/tests/test_pipeline_utils.py`

# 为什么这样做

- 当前淮安最优深度学习结果来自 `1min` 原始点预测，这个任务比新洋当前主线的 `15min` 聚合预测更难，直接对比会夸大场站差异。
- 若要做更公平的跨场站比较，最短路径不是重写训练器，而是：
  1. 先把淮安 `1min` 原始 SCADA 聚合成 canonical `15min` 主表；
  2. 再复用现有 `build_window_store.py + train_gwnet_from_store.py` 全流程。
- 这样改动面最小，也最容易与新洋 `hub_ws_only` 联合建模结果对齐。

# 做了哪些验证

- 运行：
  - `python -m pytest 15min-lead/wind/tests/test_pipeline_utils.py`
  - 结果：`13 passed`
- 运行：
  - `python -m compileall 15min-lead/wind/scripts/aggregate_scada_1min_to_15min.py 15min-lead/wind/src/xinyang_wind15/loading.py`
- 额外修正：
  - 将圆周均值 `360°` 规范化为 `0°`，避免方向均值在边界处出现等价但不一致的编码。

# 当前风险、阻塞和下一步建议

- 当前风险：
  - 新增聚合流程默认按 `timestamp.floor("15min")` 对齐窗口，等价于把每个 `15min` 标签视为该窗口起点。
  - 若原始业务定义把 `15min` 标签视为窗口终点，需要后续再核对一次和历史 `QC2` 文件的时间戳语义。
- 下一步建议：
  - 服务器端先顺序运行：
    1. `huaian_aggregate_1min_to_15min.lsf`
    2. `huaian_build_store_hubws_from_1minagg.lsf`
    3. `huaian_train_gwnet_hubws_from_1minagg.lsf`
  - 训练完成后，用 `package_remote_run.py` 把结果整理回仓库，再和：
    - 淮安 `15min` LightGBM
    - 新洋 `xinyang_gwnet_hubws_5976`
    做公平对照。

# 涉及的数据、服务器或 GitHub 操作

- 新聚合输出目标路径：
  - `/home3/s502024280003/Wind_Prediction/data/huaian/derived/ALL_TURBINES_15min_from_1min_20230703-20250703.parquet`
- 本次未执行服务器作业。
- 本次未执行 Git 提交或推送。

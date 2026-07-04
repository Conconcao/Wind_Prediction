# 本次目标

- 解释为什么当前淮安场站与新洋场站的 `15min-lead` 风速预测精度差距明显。

# 实际修改了什么

- 新增本工作报告。
- 读取并对比了以下结果摘要：
  - `15min-lead/wind/results/remote_runs/huaian_gwnet_1min_raw_ws_6308/`
  - `15min-lead/wind/results/remote_runs/huaian_tft_1min_raw_ws_6307/`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5976/`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_derived_ctx_6079/`
  - `15min-lead/wind/artifacts/local_debug/huaian_baselines_direction_full/`

# 为什么这样做

- “两个场站精度差很多”这个现象，可能来自：
  - 场站本身可预测性不同
  - 任务定义不同
  - 输入特征信息量不同
  - 模型家族不同
- 不先拆开这些因素，就会把任务难度差误判成场站差。

# 做了哪些验证

- 核对淮安当前最优远程深度学习结果：
  - `huaian_gwnet_1min_raw_ws_6308`
  - test `RMSE = 0.944643`
  - test `R2 = 0.830870`
  - 口径：`1min` 原始 SCADA、目标是 `15` 分钟后的单点 `ws`、仅 `ws`
- 核对新洋最简联合结果：
  - `xinyang_gwnet_hubws_5976`
  - test `RMSE = 0.530880`
  - test `R2 = 0.937748`
  - 口径：`15min` 聚合 `ws_mean`、仅 `ws_mean`
- 核对新洋增强结果：
  - `xinyang_gwnet_full_derived_ctx_6079`
  - test `RMSE = 0.453620`
  - test `R2 = 0.954495`
  - 比 `5976` 进一步降低 `0.077260 RMSE`，相对下降约 `14.55%`
- 核对淮安 `15min` 聚合正式基线：
  - `huaian_baselines_direction_full/lightgbm`
  - test `RMSE = 0.567931`
  - test `R2 = 0.879514`
- 关键对比：
  - 淮安从 `1min` 原始点预测切回 `15min` 聚合预测后，`RMSE` 从 `0.944643` 降到 `0.567931`
  - 下降 `0.376712`，相对下降约 `39.88%`
  - 此时它与新洋 `5976` 的差距只剩 `0.037051 RMSE`，约 `6.98%`

# 当前风险、阻塞和下一步建议

- 当前判断：
  - 现阶段两站精度差距的最大来源，不是场站本身，而是任务口径不同。
  - 淮安现在做的是更难的任务：
    - `1min` 原始序列
    - 预测未来单点值
    - 只有 `ws`
  - 新洋当前最好结果做的是更容易、信息更充分的任务：
    - `15min` 聚合风速
    - 丰富派生特征
    - 空间上下文特征
    - 联合建模 `46` 台机组
- 剩余的真实站点差异应该仍然存在，但目前只能说是次要因素，尚未被严格隔离。
- 下一步建议：
  - 若要公平比较场站，先统一口径：
    - 都做 `15min` 聚合或都做 `1min` 原始
    - 都只用 `ws`
    - 都用同一模型家族
  - 若要提升淮安结果，优先补齐：
    - 历史方向信息
    - 方向驱动的动态图支持
    - 与新洋同层级的派生特征和空间上下文

# 涉及的数据、服务器或 GitHub 操作

- 本次仅使用本地已同步结果做分析。
- 未新增服务器作业。
- 未执行 Git 提交或推送。

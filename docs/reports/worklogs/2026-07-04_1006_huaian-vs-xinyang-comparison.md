# 本次目标

- 对比当前仓库内淮安场站与新洋场站 `15min-lead` 风速预测结果。
- 区分“最优工程结果对比”和“严格可比口径对比”，避免误判。

# 实际修改了什么

- 新增本工作报告。
- 读取并核对了以下结果目录中的 `metrics.json` 与 `summary.json`：
  - `15min-lead/wind/results/remote_runs/huaian_gwnet_1min_raw_ws_6308/`
  - `15min-lead/wind/results/remote_runs/huaian_tft_1min_raw_ws_6307/`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5976/`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_derived_ctx_6079/`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/`

# 为什么这样做

- 两个场站当前最好结果的实验口径并不一致，不能只看一个 `RMSE` 就直接下结论。
- 必须先把特征集、时间分辨率、机组数量、回看窗口和预测步长核对清楚，再讨论谁更难、谁更好。

# 做了哪些验证

- 核对淮安当前最优结果：
  - `huaian_gwnet_1min_raw_ws_6308`
  - test `RMSE = 0.944643`
  - test `MAE = 0.700299`
  - test `R2 = 0.830870`
- 核对淮安最强纯时序结果：
  - `huaian_tft_1min_raw_ws_6307`
  - test `RMSE = 0.951076`
  - test `R2 = 0.828559`
- 核对新洋纯轮毂风速联合建模结果：
  - `xinyang_gwnet_hubws_5976`
  - test `RMSE = 0.530880`
  - test `MAE = 0.388906`
  - test `R2 = 0.937748`
- 核对新洋当前最优结果：
  - `xinyang_gwnet_full_derived_ctx_6079`
  - test `RMSE = 0.453620`
  - test `MAE = 0.335758`
  - test `R2 = 0.954495`
- 核对关键口径差异：
  - 淮安：`1min` 原始 SCADA，仅 `ws`，`20` 台机组，`lookback=60`，`horizon_steps=15`
  - 新洋 `5976`：`15min` SCADA，仅 `ws_mean`，`46` 台机组，`lookback=32`，`horizon_steps=1`
  - 新洋 `6079`：在 `5976` 基础上加入大量派生特征与空间上下文特征

# 当前风险、阻塞和下一步建议

- 风险：
  - 目前“淮安 vs 新洋”只能做阶段性对比，不能当成严格论文主表。
  - 淮安当前结果是 `1min raw ws-only` 口径；新洋当前最优结果是更丰富特征口径。
- 建议：
  - 如果要做公平跨场站比较，优先补齐淮安的联合建模增强版：
    - `ws + direction history`
    - 同类图模型
    - 尽量接近新洋的可用信息层级
  - 如果要做工程结论，可以先用：
    - 新洋最优：`6079`
    - 淮安最优：`6308`
  - 如果要做方法论结论，可以先用：
    - 新洋最简联合：`5976`
    - 淮安最优 `ws-only` 联合：`6308`

# 涉及的数据、服务器或 GitHub 操作

- 本次仅使用本地仓库内已同步的轻量结果文件进行分析。
- 未新增服务器作业。
- 未执行 Git 提交或推送。

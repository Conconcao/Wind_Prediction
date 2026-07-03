# 本次目标

- 按新的口径评估淮安场站 `15min-lead` 风速结果转功率后的效果：
  - 允许使用方向的历史项；
  - 不允许使用目标时刻方向特征。
- 梳理新洋场站当前效果较好的深度学习风速实验，并生成对应的淮安场站 `LSF/bsub` 训练脚本。

# 实际修改了什么

- 在 `15min-lead/wind/src/xinyang_wind15/power_curve_bridge.py` 新增了两套淮安功率曲线特征口径：
  - `HUAIAN_LATEST_HISTORY_NO_DIRECTION_*`
  - `HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_*`
- 在 `15min-lead/wind/scripts/evaluate_huaian_power_from_ws_predictions.py` 新增对应的 `--curve-feature-set` 选项：
  - `huaian_latest_history_no_direction`
  - `huaian_latest_history_past_direction_only`
- 在 `15min-lead/wind/tests/test_power_curve_bridge.py` 新增特征口径约束测试，确保：
  - 允许历史方向项；
  - 不包含目标时刻方向特征。
- 本地运行淮安 `LightGBM` 风速预测结果转功率评估，产物写入：
  - `15min-lead/wind/artifacts/local_debug/huaian_power_from_ws_predictions/lightgbm_lgbm_latest_history_past_direction_only/`
- 新增 6 个淮安场站 `LSF` 作业脚本：
  - `jobs/lsf/huaian_build_store_hubws_joint.lsf`
  - `jobs/lsf/huaian_train_gwnet_hubws_joint.lsf`
  - `jobs/lsf/huaian_build_store_directional_derived_core.lsf`
  - `jobs/lsf/huaian_train_gwnet_directional_derived_core.lsf`
  - `jobs/lsf/huaian_build_store_directional_derived_ctx.lsf`
  - `jobs/lsf/huaian_train_gwnet_directional_derived_ctx.lsf`

# 为什么这样做

- 之前 WTPC 现成的“最新版”淮安多变量功率曲线结果，要么：
  - 含目标时刻方向特征；
  - 要么不含历史项。
- 这与当前要求不一致，所以不能直接套已有结果，必须在本地重新定义并评估一套严格满足约束的口径。
- 新洋场站当前最强的深度学习结果集中在 `GWNet` 线上：
  - `full_derived_ctx`
  - `full_derived_core`
  - `hubws_joint`
- `AGCRN / MTGNN / ModernTCN` 当前在新洋 `15min-lead` 上明显弱于上述 `GWNet` 结果，因此不优先迁移到淮安，避免浪费算力。
- 淮安目前没有塔筒气象输入，因此“对应迁移”采用的是同类思路而非逐字段照搬：
  - `hubws_joint`：保留纯轮毂风速联合建模基线；
  - `directional_derived_core`：方向 + 1min 聚合 + 派生特征；
  - `directional_derived_ctx`：在上一版基础上再引入空间上下文。

# 做了哪些验证

- 单元测试：
  - `python -m pytest 15min-lead/wind/tests/test_power_curve_bridge.py`
  - 结果：`8 passed`
- 本地功率曲线评估：
  - 输入风速预测：
    - `15min-lead/wind/artifacts/local_debug/huaian_baselines_direction_full/lightgbm_val_predictions.csv`
    - `15min-lead/wind/artifacts/local_debug/huaian_baselines_direction_full/lightgbm_test_predictions.csv`
  - 输出目录：
    - `15min-lead/wind/artifacts/local_debug/huaian_power_from_ws_predictions/lightgbm_lgbm_latest_history_past_direction_only/`
- 关键结果：
  - 风速侧最佳已用结果：`LightGBM test rmse_macro = 0.5676512746135893`
  - 功率侧 `test nRMSE = 0.09255448751896808`
  - 功率侧 `test 合格率 Q = 93.74150995075948`
  - 功率曲线上限 sanity：`test nRMSE = 0.03678264702359842`，`Q = 97.40701469310477`
- 与更严格的“完全不含方向信息历史项”版本相比：
  - `test nRMSE` 从 `0.09266719645219362` 降到 `0.09255448751896808`
  - `test Q` 从 `93.73584954255364` 升到 `93.74150995075948`
  - 说明加入“历史方向项”有小幅正增益，但不是决定性增益。
- 脚本语法检查：
  - 对新增 6 个 `jobs/lsf/*.lsf` 运行了 `bash -n`
  - 结果通过；本机 `WSL` 输出了若干本地 `localhost/NAT` 警告，但未影响语法检查结果。

# 当前风险、阻塞和下一步建议

- 当前风险：
  - 淮安暂无完整深度学习正式结果，现阶段功率评估仍基于本地 `LightGBM` 风速结果。
  - `directional_derived_core` / `directional_derived_ctx` 是“按淮安可用数据适配后的对应方案”，不是对新洋最优实验的逐字段复刻。
  - WTPC 历史特征构造在本地评估时会产生大量 `PerformanceWarning`，目前不影响结果正确性，但后续可优化速度。
- 下一步建议：
  - 先在服务器端依次跑：
    - `huaian_store_hubws_joint` + `huaian_train_gwnet_hubws_joint`
    - `huaian_build_store_directional_derived_core` + `huaian_train_gwnet_directional_derived_core`
    - `huaian_build_store_directional_derived_ctx` + `huaian_train_gwnet_directional_derived_ctx`
  - 等淮安深度学习风速结果回仓库后，再统一代入当前这套“历史含方向、目标时刻不含方向”的功率曲线口径，比较：
    - 风速提升是否能稳定转化为功率合格率提升；
    - `ctx` 是否在淮安也能带来可观增益。

# 涉及的数据、服务器或 GitHub 操作

- 本地读取数据与结果：
  - `C:\Users\caosh\Desktop\WTPC`
  - `15min-lead/wind/artifacts/local_debug/huaian_baselines_direction_full/`
- 本轮新增的服务器脚本默认都使用：
  - 项目根：`/home3/s502024280003/Wind_Prediction`
  - 调度器：`LSF / bsub`
  - 每次作业：`1` 块 GPU
- GitHub 侧：
  - 本轮只应提交与淮安功率评估口径、测试和新增 `LSF` 作业脚本直接相关的文件。

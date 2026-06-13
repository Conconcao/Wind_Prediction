## 本次目标

- 把前一轮收敛出来的两类“可直接派生”特征真正接入新洋 `15min` 管线：
  - 第 1 组：局地物理/风廓线派生特征
  - 第 2 组：方向感知上风向空间上下文特征

## 实际修改了什么

- 在 `15min-lead/wind/src/xinyang_wind15/features.py` 中新增并接入：
  - `add_operational_proxy_features`
    - `derived_ti_15m`
    - `derived_gust_factor_15m`
    - `derived_gust_excess_15m`
    - `derived_ws_range_15m`
  - `add_tower_profile_features`
    - `profile_shear_alpha_10m_125m`
    - `profile_shear_alpha_70m_125m`
    - `profile_veer_10m_125m_*`
    - `profile_veer_70m_125m_*`
    - `profile_temperature_delta_125m_10m`
    - `profile_pressure_delta_125m_10m`
    - `hub_tower_ws_125m_delta`
    - `hub_tower_wd_125m_*`
  - `add_direction_aware_spatial_context`
    - `ctx_upwind_ws_mean`
    - `ctx_upwind_power_mean`
    - `ctx_upwind_ws_gap`
    - `ctx_upwind_power_gap`
    - `ctx_upwind_weight_sum`
    - `ctx_upwind_count`
    - `ctx_upwind_nearest_dist_km`
    - `ctx_crosswind_nearest_abs_km`
- 扩展 `build_timestep_feature_frame` 和 `build_supervised_frame`：
  - 支持 `include_derived_core`
  - 支持 `include_spatial_context`
  - 支持空间上下文超参数
- 在 `15min-lead/wind/src/xinyang_wind15/graph.py` 中新增 `build_distance_matrix`，复用于方向感知空间特征。
- 在 `15min-lead/wind/src/xinyang_wind15/feature_presets.py` 中新增特征块选择器：
  - `tower`
  - `one_min`
  - `derived_core`
  - `spatial_context`
- 更新脚本入口：
  - `15min-lead/wind/scripts/build_window_store.py`
  - `15min-lead/wind/scripts/build_window_dataset.py`
  - 新增参数：
    - `--include-derived-core`
    - `--include-spatial-context`
    - `--spatial-direction-sigma-deg`
    - `--spatial-distance-scale-km`
- 更新测试：
  - `15min-lead/wind/tests/test_feature_presets.py`
  - `15min-lead/wind/tests/test_pipeline_utils.py`
- 更新文档：
  - `15min-lead/wind/README.md`

## 为什么这样做

- 之前的 `6058` 全量模型虽然效果最好，但它主要使用的是“原值 + 1min 聚合原值”，还没有把物理上更直接的结构信息显式编码出来。
- 这次实现遵循两个原则：
  - 先做完全依赖现有数据即可派生的特征，不引入外部数据依赖
  - 以“可开关、可消融”的方式接入，而不是直接改默认主线配置
- 对边界机组的空间上下文，不再返回缺失值，而是返回显式中性回退：
  - `count = 0`
  - `weight_sum = 0`
  - 上风向均值回退到本机当前值
  - gap 回退到 `0`
  这样比简单前向填充更符合问题本质，也避免 dense-window 构建时被缺失值整体淘汰。

## 做了哪些验证

- 单元测试：
  - `pytest 15min-lead/wind/tests -q`
  - 结果：`25 passed`
- 新增测试覆盖了：
  - 风廓线 shear/veer 派生是否按预期计算
  - `hub vs tower` 失配特征是否正确
  - 上风向上下文方向约定是否正确
  - 特征块选择器是否按前缀工作
- 脚本 smoke 测试：
  - `build_window_store.py`
    - `--include-tower --include-derived-core --include-spatial-context`
    - `4` 台机组、`64` 个时间点
    - 成功落盘
  - `build_window_dataset.py`
    - 同配置下 `64` 个时间点因该子集无足够有效窗口未通过
    - 扩大到 `512` 个时间点后成功生成窗口

## 当前风险、阻塞和下一步建议

- 风险：
  - 当前空间上下文仍是“单时刻、当前风向驱动”的静态派生特征，还没有显式建模传播时滞。
  - 风廓线和上风向上下文都已进管线，但现在还没有对应的服务器端全场消融结果。
- 阻塞：
  - 尚未新增专门的 `LSF` 作业脚本来跑这两组新特征的全场 `GWNet` 消融。
- 下一步建议：
  1. 先基于 `GWNet 6058` 这条主线做两组增量 store：
     - `scada_core + tower + derived_core`
     - `scada_core + tower + derived_core + spatial_context`
  2. 先不改模型结构，只比较特征增益
  3. 若这两组确有稳定增益，再决定是否继续引入传播时滞或动态图结构

## 涉及的数据、服务器或 GitHub 操作

- 本轮未改动原始数据。
- 本轮只在本地读取了：
  - `15min` SCADA
  - 塔架观测
  - 风机元数据
- 本轮未提交服务器作业。
- 本轮未推送 GitHub。

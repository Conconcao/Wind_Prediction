# 2026-06-12 Direction and Yaw Feature Strategy

## 本次目标

- 回答“只加入风向和机组朝向是否有效”以及“怎样更好利用这些信息”。
- 区分当前代码里已经做了什么，和下一步真正值得做的方向。

## 实际修改了什么

- 检查了当前方向特征相关代码：
  - `15min-lead/wind/src/xinyang_wind15/features.py`
  - `15min-lead/wind/src/xinyang_wind15/feature_presets.py`
  - `15min-lead/wind/scripts/build_window_store.py`
- 检查了当前 `6058` 结果的特征摘要：
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/store_summary.json`
- 检索并阅读了与风向、机组朝向、动态图和短时预测相关的文献摘要与原文页面。

## 为什么这样做

- 风向和机组朝向是否有用，不能只看直觉。
- 这个问题既取决于风电场物理机理，也取决于当前特征编码方式是否合理。

## 做了哪些验证

- 当前代码在特征工程阶段已经构造了方向的圆周编码：
  - `wd -> wd_sin / wd_cos`
  - `nacelle_mean -> nacelle_sin / nacelle_cos`
- 但当前 `default_multivariate` 预设并没有把这些圆周编码放进最终 store，
  主要保留的是原始 `wd_mean` 与 `nacelle_mean`。
- 文献结论与当前任务高度相关：
  - 动态图短时风电预测工作明确把“历史风速 + 风向”作为图结构随时间变化的核心驱动。
  - 机组控制与风场物理研究表明，风向变化和偏航失配会改变尾流传输与机组间影响关系。
  - 仅使用当前时刻风速与风向的简化模型通常弱于包含时间序列上下文的模型。
  - 最小特征子集研究通常仍保留风向，但不会只保留风向而去掉风速量级信息。

## 当前结论

- 只加入风向和机组朝向，大概率会有提升，但通常是“小到中等增益”，
  不太可能单独达到 `6058` 这种全特征结果。
- 更重要的不是“有没有加方向”，而是“怎么加”：
  - 原始角度直接入模不理想，圆周编码更合理。
  - 风向与机组朝向的相对角度（偏航失配）往往比两者各自原值更有信息量。
  - 对全场联合建模来说，方向信息最应该体现在动态图或上风向聚合上，而不只是节点特征。
- 若只利用“风向信息 + 历史状态信息”，这是一个很值得做的紧凑版方案；
  但若真的只留风向、不留风速幅值信息，预测力大概率会明显不足。

## 当前风险、阻塞和下一步建议

- 风险：
  - 把角度当普通实数输入，容易产生 `359°` 与 `1°` 被模型误判为“差很大”的问题。
- 阻塞：
  - 当前仓库里还没有专门的“方向/朝向/偏航失配”消融配置与作业脚本。
- 下一步建议：
  - 先做最短路径实验：
    - `ws_mean`
    - `ws_mean + wd_sin/cos`
    - `ws_mean + wd_sin/cos + nacelle_sin/cos`
    - `ws_mean + wd_sin/cos + nacelle_sin/cos + yaw_error`
  - 然后再升级到：
    - 基于当前风向的上风向动态图
    - 上游扇区聚合特征
    - 按主风向分扇区的 mixture-of-experts

## 涉及的数据、服务器或 GitHub 操作

- 本轮未修改原始数据。
- 本轮未提交服务器作业。
- 本轮未推送 GitHub，仅新增本地工作报告。

## 参考文献

- Fan et al., 2021, Dynamic directed spatiotemporal graph neural network for wind power forecasting, arXiv:2108.13285
- Dallas et al., 2024, The role of preview wind direction measurements on utility-scale wind turbine energy production and yaw misalignment, Wind Energy Science
- Ally et al., 2025, A modular deep-learning approach for wind farm power forecasting in an operating offshore wind farm, Wind Energy Science
- Tautz-Weinert et al., 2025 preprint, Minimum data set for data-driven wind power forecasting at turbine scale, Wind Energy Science Discussions

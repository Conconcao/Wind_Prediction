## 本次目标

- 为“`6058` 主线 + 直接派生特征”的下一轮增量实验补齐专用 `LSF` 作业脚本与服务器端命令。

## 实际修改了什么

- 新增 `LSF` store 构建脚本：
  - `jobs/lsf/xinyang_build_store_full_derived_ablation.lsf`
- 新增 `LSF` `GWNet` 训练脚本：
  - `jobs/lsf/xinyang_train_gwnet_full_derived_ablation.lsf`
- 更新 `15min-lead/wind/README.md`：
  - 把两份新脚本加入 server-side file 列表
  - 新增“Full-plus-derived ablations”说明
  - 补充两组单行 `bsub` 命令
  - 补充结果打包单行命令
- 本地做了 `6058` 主线超集 smoke：
  - `default_multivariate + tower + 1min + derived_core + spatial_context`

## 为什么这样做

- 现在最关键的问题已经不是“直接派生特征本身是否有效”，而是：
  - 它们在当前最强主线 `6058` 上是否还有边际增益
- 现有 `direct_derived` 作业脚本虽然技术上能复用，但默认是轻量 `scada_core` 线。
- 这轮新增的脚本把 `6058` 主线骨架直接固定下来：
  - `feature_preset=default_multivariate`
  - `include_tower=1`
  - `include_1min=1`
  - `include_derived_core=1`
- 这样用户在服务器端只需切换：
  - `INCLUDE_SPATIAL_CONTEXT=0`
  - 或 `INCLUDE_SPATIAL_CONTEXT=1`
  就能跑两组增量实验，避免手工拼装错误。

## 做了哪些验证

- 使用本地真实数据做了 store smoke：
  - `4` 台机组
  - `12000` 个时间点
  - `217` 个输入特征
  - 成功生成 store，且覆盖 `train/val/test`
- 在同一 smoke store 上运行了 `GWNet` `1 epoch` 训练：
  - 训练脚本成功启动并完成
  - 说明“`6058` 主线 + derived + ctx”组合在现有代码路径上是可执行的
- 对两份新 `LSF` 脚本执行了 `bash -n` 语法检查；
  本地 `WSL` 输出有噪声，但命令返回码为 `0`，未发现脚本语法错误。

## 当前风险、阻塞和下一步建议

- 风险：
  - 这轮只验证了小样本本地 smoke，服务器整年全场作业仍可能因为时长或显存需要再微调。
- 阻塞：
  - 需要用户在服务器端实际提交新作业并返回作业号。
- 下一步建议：
  1. 先跑 `full_derived_core`
  2. 再跑 `full_derived_ctx`
  3. 完成后打包进仓库并做与 `6058` 的严格增量比较

## 涉及的数据、服务器或 GitHub 操作

- 本轮未修改原始数据。
- 本轮未提交服务器作业，只补齐了脚本和命令说明。
- 本轮未推送 GitHub。

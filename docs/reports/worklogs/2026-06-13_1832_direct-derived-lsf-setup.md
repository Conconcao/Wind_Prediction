## 本次目标

- 为新洋 `15min` 的两组“可直接派生特征”实验补齐服务器端 `LSF/bsub` 入口和结果回仓库命令。

## 实际修改了什么

- 新增参数化 `LSF` store 构建脚本：
  - `jobs/lsf/xinyang_build_store_direct_derived_ablation.lsf`
- 新增参数化 `LSF` `GWNet` 训练脚本：
  - `jobs/lsf/xinyang_train_gwnet_direct_derived_ablation.lsf`
- 更新 `15min-lead/wind/README.md`：
  - 将这两份新脚本加入 server-side file 列表
  - 新增“Server-side direct-derived ablations”小节
  - 写入两组推荐 `STORE_TAG / RUN_TAG`
  - 写入 `bsub` 单行提交示例
  - 写入 `package_remote_run.py` 单行打包示例

## 为什么这样做

- 这两组实验本质上只有少量开关差异：
  - 是否开启 `spatial_context`
  - store/run 目录标签不同
- 因此最短路径不是复制两套几乎一样的脚本，而是做一对参数化模板，用环境变量切换。
- 这样后续如果你还要扩展：
  - `+ include_1min`
  - 只开 `derived_core`
  - 调整空间方向核宽度
  都不用再复制新作业脚本。

## 做了哪些验证

- 核对了现有 `full_validfix`、`direction_ablation` 作业脚本风格，保证新脚本与当前仓库工作流一致。
- 检查了新脚本中的：
  - 项目根目录
  - conda 激活逻辑
  - `output/log/store/run` 路径
  - 环境变量默认值
  - `build_window_store.py` / `train_gwnet_from_store.py` 参数名
- 用 `bash -n` 对两份新增 `LSF` 脚本做了语法检查，未发现脚本语法错误。

## 当前风险、阻塞和下一步建议

- 风险：
  - 当前只是把服务器端入口补齐了，还没有实际提交并验证全场作业资源占用是否需要进一步调参。
- 阻塞：
  - 需要用户在服务器端实际提交这两组作业，并把作业号返回。
- 下一步建议：
  1. 先提交 `derived_core`
  2. 再提交 `derived_ctx`
  3. 训练完成后用 README 里的单行命令打包结果并 push
  4. 本地拉取后优先分析相对 `5976/6058` 的增益来源

## 涉及的数据、服务器或 GitHub 操作

- 本轮未改动原始数据。
- 本轮未实际提交服务器作业，只新增了 `LSF` 作业脚本和命令说明。
- 本轮未推送 GitHub。

# 2026-06-12 Xinyang GWNet Full Validfix 6058 Local Analysis

## 本次目标

- 同步并核对服务器回传后已推送到仓库的 `6058` 结果。
- 判断修复后的多变量 `GWNet` 是否真正优于当前主对照
  `xinyang_gwnet_hubws_5976`。
- 给出是否将 `6058` 作为当前 `15min` 主结果的结论。

## 实际修改了什么

- 本地执行了 `git pull` 后的结果核对，确认当前仓库头指针为：
  - `17dd43f Add xinyang multivariate validfix run 6058 summaries`
- 读取并比对了以下结果文件：
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/metrics.json`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/summary.json`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/test_per_turbine.csv`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5976/metrics.json`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5976/test_per_turbine.csv`
- 新增本报告，记录本轮结论和依据。

## 为什么这样做

- `6058` 是修复有效窗口逻辑后的多变量全特征重跑版本，只有把它和
  `5976` 做一对一比较，才能回答“更宽特征集是否真的有用”这个核心问题。
- 如果只看单次作业是否跑通，无法判断空间联合建模与多变量特征是否值得保留。
- 先给出定量结论，再决定后续是否继续扩展到更多模型或做消融，路径更短。

## 做了哪些验证

- 核对 `6058` 测试集总体指标：
  - `RMSE = 0.456015`
  - `MAE = 0.338033`
  - `R2 = 0.954581`
- 与 `5976` (`hub_ws_only` joint `GWNet`) 对比：
  - `RMSE` 从 `0.531345` 降到 `0.456015`
  - 绝对下降 `0.075330`
  - 相对下降约 `14.18%`
  - `MAE` 下降 `0.050804`
  - `R2` 提升 `0.016245`
  - `macro RMSE` 下降 `0.075287`
  - `macro R2` 提升 `0.016292`
- 核对逐风机测试结果：
  - `46 / 46` 台机组的 `RMSE` 全部优于 `5976`
  - 改善最大的机组包括：
    - `S36: 0.577323 -> 0.478537`
    - `S40: 0.570621 -> 0.473962`
    - `S43: 0.551446 -> 0.457751`
    - `S09: 0.530158 -> 0.437105`
    - `S39: 0.566789 -> 0.476240`
- 核对 `6058` 配置摘要：
  - `181` 个输入特征
  - `18926 / 6032 / 2941` 个 train/val/test windows
  - `supports = 2`
  - `support_mode = distance_correlation`
  - `adaptive_adj = true`

## 当前结论

- `6058` 不是边际改进，而是显著且稳定优于当前 `hub_ws_only` 主结果。
- 现阶段应把 `xinyang_gwnet_full_validfix_6058` 视为新洋 `15min`
  全场联合风速预测的当前最佳结果。
- 这说明：
  - 空间联合建模有效
  - 仅用轮毂高度风速做纯时序外推不是当前最优方案
  - 塔架变量、机舱/功率统计量和缺测指示在当前数据条件下带来了真实增益

## 当前风险、阻塞和下一步建议

- 风险：
  - 目前最强结果仍主要来自 `GWNet` 家族，尚未完成系统性消融，无法精确分离
    “图结构收益”和“额外特征收益”各自贡献。
- 阻塞：
  - 无直接阻塞；结果已可用于后续分析和汇报。
- 下一步建议：
  - 以 `6058` 作为主结果基线。
  - 做一轮特征消融，至少比较：
    - `hub_ws_only`
    - `hub_ws + tower`
    - `hub_ws + tower + nacelle/power`
  - 再补一轮公平对照：
    - `AGCRN` 多变量版本
    - `MTGNN` 多变量版本
  - 若目标转向论文或汇报，优先开始整理：
    - 主表格
    - 每机组提升分布图
    - 误差分位数或风速区间分层分析

## 涉及的数据、服务器或 GitHub 操作

- GitHub：
  - 拉取了用户已推送的 `6058` 结果摘要提交。
- 服务器：
  - 本轮未重新提交作业，仅分析已回传并入库的结果。
- 数据：
  - 本轮未新增或改写原始数据，只读取仓库中的轻量结果摘要文件。

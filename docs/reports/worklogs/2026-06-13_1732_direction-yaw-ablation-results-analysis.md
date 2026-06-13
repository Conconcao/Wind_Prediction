# 2026-06-13 Direction Yaw Ablation Results Analysis

## 本次目标

- 同步并分析服务器端已推送的方向/朝向消融实验结果：
  - `6066` `D1 = direction_wd_only`
  - `6068` `D3 = direction_wd_yaw_error`
  - `6069` `D4 = direction_wd_yaw_error + dynamic directional support`
- 判断风向、偏航失配和动态图方向 support 是否在当前数据上带来稳定增益。

## 实际修改了什么

- 本地执行 `git pull --rebase origin main`，同步到：
  - `e51aa1a`
- 读取并比对了以下结果目录：
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_direction_wd_6066`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_direction_wd_yaw_error_6068`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_direction_wd_yaw_error_dyn_6069`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5976`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058`
- 新增本报告，记录结果解读和下一步建议。

## 为什么这样做

- 方向/朝向这条线的核心问题不是“能不能跑”，而是“值不值得继续投入”。
- 必须把 `D1/D3/D4` 和当前纯时序联合基线 `5976` 以及最强多变量结果 `6058` 放在一起看，才能判断方向信息的真实边际价值。

## 做了哪些验证

- 核对 `summary.json`，确认三组新实验都保持了与 `5976` 可比的主设置：
  - `18926 / 6032 / 2941` train/val/test windows
  - `support_mode = distance_correlation`
  - `adaptive_adj = true`
- 核对特征配置：
  - `6066` 使用 `ws_mean + wd_sin + wd_cos`
  - `6068` 使用 `ws_mean + wd_sin/cos + nacelle_sin/cos + yaw_error_sin/cos/abs`
  - `6069` 在 `6068` 基础上额外启用动态方向 support，`supports = 4`
- 测试集总体指标对比：
  - `5976 (D0)`: `RMSE = 0.531345`, `MAE = 0.388838`, `R2 = 0.938336`
  - `6066 (D1)`: `RMSE = 0.528993`, `MAE = 0.389106`, `R2 = 0.938880`
  - `6068 (D3)`: `RMSE = 0.529660`, `MAE = 0.389152`, `R2 = 0.938726`
  - `6069 (D4)`: `RMSE = 0.531290`, `MAE = 0.388149`, `R2 = 0.938348`
  - `6058 (full multivariate)`: `RMSE = 0.456015`, `MAE = 0.338033`, `R2 = 0.954581`
- 相对 `5976` 的变化：
  - `D1`: `RMSE -0.002352`, `R2 +0.000545`
  - `D3`: `RMSE -0.001685`, `R2 +0.000390`
  - `D4`: `RMSE -0.000054`, `R2 +0.000013`
- 逐机组改善台数相对 `5976`：
  - `D1`: `32 / 46`
  - `D3`: `31 / 46`
  - `D4`: `20 / 46`
- 训练历史最佳验证轮次：
  - `D1`: best epoch `28`
  - `D3`: best epoch `24`
  - `D4`: best epoch `24`

## 当前结论

- 只加入风向圆周编码是有效的，但增益很小。
- 在当前数据和当前实现下，加入机组朝向与偏航失配没有进一步带来稳定提升。
- 动态方向 support 这版最小实现基本没有带来额外收益，说明：
  - 方向信息本身有一些价值
  - 但当前“按最新一步风向重排图结构”的方法还不够强，或噪声较大
- `D1/D3/D4` 全部远弱于 `6058`，说明方向/朝向特征不能替代更丰富的多变量信息。

## 当前风险、阻塞和下一步建议

- 风险：
  - 若继续沿方向线大幅加码，可能只是在追逐很小的边际改进。
- 阻塞：
  - 当前没有直接阻塞，结果已经足够支持决策。
- 下一步建议：
  - 若目标是尽快提升性能，优先回到 `6058` 主线做特征消融，而不是继续深挖 `D4`。
  - 若目标是论文或机理解释，可保留 `D1` 作为一个“方向信息确有帮助”的紧凑对照。
  - 暂不建议继续扩大 `D3/D4`，除非准备同时引入：
    - 更明确的上游扇区聚合
    - 传播时滞
    - 更物理的尾流约束

## 涉及的数据、服务器或 GitHub 操作

- GitHub：
  - 拉取了用户从服务器端推回的 `6066/6068/6069` 结果摘要提交。
- 服务器：
  - 本轮未重新提交作业，仅对已入库结果做本地分析。
- 数据：
  - 本轮未修改原始数据，只读取仓库中的结果摘要文件。

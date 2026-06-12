# 2026-06-12 Feature Gain Attribution Check

## 本次目标

- 判断当前 `6058` 相对 `5976` 的提升，已经能够归因到什么粒度。
- 区分“现在已经知道的原因”和“必须再做实验才能知道的原因”。

## 实际修改了什么

- 读取并核对了以下文件：
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5976/summary.json`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/summary.json`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/store_summary.json`
  - `15min-lead/wind/src/xinyang_wind15/feature_presets.py`
  - `15min-lead/wind/src/xinyang_wind15/features.py`
  - `15min-lead/wind/src/xinyang_wind15/loading.py`
- 新增本报告，记录当前归因结论。

## 为什么这样做

- `6058` 的提升很明显，但“提升来自哪些特征”不能靠印象回答。
- 先确认两次运行到底改变了什么，才能决定下一步做解释性分析还是消融实验。

## 做了哪些验证

- 确认 `5976` 与 `6058` 使用的是同类 `GWNet` 联合建模路径，且：
  - `supports = 2`
  - `support_mode = distance_correlation`
  - `adaptive_adj = true`
  - `18926 / 6032 / 2941` 个 train/val/test windows
- 确认两者关键差异主要在输入特征：
  - `5976`：仅 `ws_mean`
  - `6058`：`181` 个特征
- 将 `6058` 的 `181` 个特征按来源归组后，得到：
  - `1` 个基础目标同源特征：`ws_mean`
  - `4` 个额外 `15min` SCADA 特征：
    - `ws_std`
    - `wd_mean`
    - `power_mean`
    - `nacelle_mean`
  - `26` 个塔架相关特征：
    - `13` 个塔架气象原值
    - `13` 个对应缺测指示
  - `150` 个 `1min` 聚合相关特征：
    - `75` 个聚合统计量
    - `75` 个对应缺测指示

## 当前结论

- 现在已经能确定：
  - `6058` 的提升来自特征扩充，而不是模型结构、图支撑方式或样本切分变化。
- 现在还不能确定：
  - 到底是 `tower`、`1min` 聚合、`15min` 额外 SCADA 变量中的哪一组贡献最大。
  - 也不能从现有两次结果直接判断某个单独特征的边际贡献。
- 物理上最可能的主贡献来源是：
  - `1min` 风速/功率/机舱角的 `15/30/60min` 聚合统计与 ramp
  - 多高度塔架风速/风向
  - `wd_mean`、`ws_std`、`power_mean`、`nacelle_mean`
- 缺测指示大概率主要提供稳定性和数据质量上下文，而不是主要提升来源。

## 当前风险、阻塞和下一步建议

- 风险：
  - 直接做单特征解释会把“模型当前依赖什么”与“哪些特征真正带来提升”混为一谈。
- 阻塞：
  - 仓库里没有模型权重，只保留了结果摘要，因此无法直接对现有 `6058` 做本地扰动解释。
- 下一步建议：
  - 若目标是回答“哪些特征带来了提升”，优先做固定模型下的分组消融重训。
  - 若目标是回答“当前 `6058` 模型最依赖哪些输入”，则在服务器保留权重后做分组 permutation importance。

## 涉及的数据、服务器或 GitHub 操作

- 本轮未改动原始数据。
- 本轮未提交服务器新作业。
- 本轮未推送 GitHub，仅做本地代码与结果文件核对。

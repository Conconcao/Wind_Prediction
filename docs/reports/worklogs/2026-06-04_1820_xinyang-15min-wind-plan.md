# 2026-06-04 新洋 15 分钟风速实验方案报告

## 本次目标

- 围绕 `xinyang` 风电场设计 `15-minute-ahead` 风速预测实验。
- 检索并阅读相关文献，结合本地数据条件形成一套可执行方案。
- 在 `15min-lead/wind/` 下建立本轮工作的目录和配置骨架。

## 本次完成内容

- 检查了 `15min-lead/wind/`，并新建了以下子目录：
  - `docs/`
  - `configs/`
  - `scripts/`
  - `references/`
  - `artifacts/`
- 盘点并确认了新洋站点可用数据：
  - `15min` SCADA 主表
  - `1min` SCADA 主表
  - 多层高气象观测表
  - 风机元数据表
  - 可选的开源 GFS NWP 数据
- 固定了按唯一时间戳的 `7:2:1` 时序切分边界：
  - train:
    `2025-01-01 00:15:00` 到 `2025-09-15 16:00:00`
  - val:
    `2025-09-15 16:15:00` 到 `2025-11-26 08:45:00`
  - test:
    `2025-11-26 09:00:00` 到 `2026-01-01 00:00:00`
- 编写了以下实验文档：
  - `15min-lead/wind/README.md`
  - `15min-lead/wind/docs/xinyang_15min_wind_literature_and_plan.md`
  - `15min-lead/wind/references/papers.md`
  - `15min-lead/wind/configs/splits/xinyang_7_2_1.yaml`
  - `15min-lead/wind/configs/models/model_shortlist.yaml`

## 文献检索与采用理由

本次重点参考了以下文献方向：

- 综述类：
  - 深度学习在风速/风电预测中的总体发展趋势
  - 短时风速预测中高分辨率观测与模型结构的关系
- 基线类：
  - LSTM、GRU 在短时风速预测中的表现
  - SARIMA 与 GRU/LSTM 的对比
- 时空建模类：
  - 多站点联合学习
  - 图注意力、多图结构和动态图结构在风场预测中的应用
- Transformer 类：
  - 多变量风速预测中的 Transformer 结构

最终结论是：

- 以深度学习为主是合理的；
- 但对新洋这类 `46` 台风机、同一风场、单年数据的任务，图时空模型比直接上 Transformer 更匹配数据结构；
- Transformer 适合作为后续 ablation，而不是第一优先模型；
- persistence、seasonal persistence、SARIMA、LightGBM 应保留为本地基线。

## 方案核心结论

- 主任务定义：
  - 以 `46` 台风机为节点，预测下一步 `15 min` 的 `平均风速`
- 主特征路线：
  - 先做 `15min SCADA`
  - 再加时间编码
  - 再加多层高观测塔气象特征
  - 最后再考虑 `1min` 聚合特征和可选 NWP
- 模型优先级：
  - 本地基线：
    persistence、seasonal persistence、SARIMA、LightGBM
  - 深度学习基线：
    GRU、TCN
  - 主模型：
    GraphWaveNet 风格或 AG-MGAT 风格的图时空网络
  - 后续对照：
    WindFormer 或 PatchTST 风格 Transformer

## 为什么这样做

- 新洋数据已经足够支撑严格的短期时序建模，不必一开始就依赖外部 NWP。
- 15 分钟主表、1 分钟高频表和多层高气象观测组合在一起，适合逐步做 feature ablation。
- 图时空网络可以直接利用风机间的地理布局和相关性，最符合风场多机组数据结构。
- 先把轻量基线在本地跑通，再推进深度学习主模型，更符合本项目“先本地开发验证，再推 GitHub，再上超算”的规则。

## 验证情况

- 已确认 `xinyang` 共有 `46` 台风机。
- 已确认 15 分钟主表有 `34,208` 个唯一时间戳，目标列 `平均风速` 无缺失。
- 已确认观测塔层高为 `10/30/50/70/125 m`。
- 已确认观测塔时间覆盖约为主表时间戳的 `97.39%`。
- 已确认开源 NWP 只有 `2025-07-01` 之后的数据，且为逐小时数据，因此仅建议作为后续可选 ablation。

## 下一步建议

- 直接开始在 `15min-lead/wind/scripts/` 中实现：
  - 数据读取与统一字段映射
  - 时序切分逻辑
  - persistence / seasonal persistence / SARIMA / LightGBM 基线
- 然后再实现：
  - GRU
  - TCN
  - 图时空主模型


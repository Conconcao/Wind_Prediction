## 本次目标

- 在“查阅文献看下还有哪些信息对超短期风速预测有增益”的基础上，进一步收窄到“只保留当前新洋数据中可直接派生、可立即落地实验”的特征方向。

## 实际修改了什么

- 读取并核对了当前 `15min` 实验的数据字段与特征管线：
  - `15min-lead/wind/src/xinyang_wind15/schema.py`
  - `15min-lead/wind/src/xinyang_wind15/features.py`
  - `15min-lead/wind/results/remote_runs/xinyang_gwnet_full_validfix_6058/summary.json`
  - `15min-lead/wind/docs/xinyang_15min_wind_literature_and_plan.md`
- 新增本报告，记录“可直接派生特征”的候选清单与优先级。

## 为什么这样做

- 上一轮文献结论里既有“当前数据就能做”的方向，也有“需要外部数据或新传感器”的方向。
- 如果不先做这一步收窄，后续很容易把实验时间花在暂时不可执行的方案上。
- 先按当前数据约束筛一遍，能让下一步代码实现直接对准高价值、低依赖的特征组。

## 做了哪些验证

- 确认当前原始数据中可直接使用的来源包括：
  - `15min` 机组级 SCADA：`ws/power/nacelle/wd` 及统计量
  - `1min` 机组级 SCADA：`ws/power/nacelle_angle/wd`
  - 塔架 `10/30/50/70/125m`：`temperature/humidity/ws/wd/pressure`
  - 风机元数据：经纬度与机组布局
- 确认当前 `6058` 全量模型已包含：
  - `1min` 聚合统计
  - 多高度塔架原值
  - `ws_mean / ws_std / wd_mean / power_mean / nacelle_mean`
- 确认当前尚未系统显式派生、但可立即新增实验的高价值方向主要是：
  - 垂直风廓线派生量：`shear / veer / lapse / hub-tower mismatch`
  - 方向感知空间派生量：`upwind weighted context / along-wind and cross-wind geometry`
  - 局地波动与湍流代理：`TI / gust factor / ramp / direction concentration`
  - 机组控制相对量：`rolling yaw error` 及其波动统计

## 当前风险、阻塞和下一步建议

- 风险：
  - 直接把所有可派生特征一次性塞进模型，会再次造成“有效但不知道为什么有效”。
- 阻塞：
  - 当前没有分组消融结果，暂时不能判断哪一组派生特征最值得优先工程化。
- 下一步建议：
  1. 先做一组“小而强”的新增特征：`shear + veer + TI + gust + hub/tower mismatch`
  2. 再做一组“方向感知空间特征”：`upwind weighted ws/power + along/cross wind distance`
  3. 采用固定主模型（建议 `GWNet 6058` 这条线）做分组消融，而不是同时改模型结构

## 涉及的数据、服务器或 GitHub 操作

- 本轮未修改原始数据。
- 本轮未提交服务器作业。
- 本轮未推送 GitHub，仅做本地代码与结果核对，并新增工作报告。

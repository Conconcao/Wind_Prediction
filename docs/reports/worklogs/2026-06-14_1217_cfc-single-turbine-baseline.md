# 本次目标

为新洋 `15min-lead` 风速预测项目新增一条最小可跑的单机 `CfC` 对照实验路线，用来判断液态/连续时间风格模型是否值得进入后续主仓实验。

## 实际修改了什么

- 新增 `CfC` 模型封装：
  - `15min-lead/wind/src/xinyang_wind15/cfc.py`
- 新增单机/多变量兼容的本地训练脚本：
  - `15min-lead/wind/scripts/train_cfc_baseline.py`
- 新增服务器端 `LSF` 单机任务模板：
  - `jobs/lsf/xinyang_train_cfc_single_turbine.lsf`
- 更新了以下工程文件：
  - `15min-lead/wind/requirements.txt`
  - `15min-lead/wind/README.md`
  - `15min-lead/wind/configs/models/model_shortlist.yaml`
  - `15min-lead/wind/references/implementation_sources.md`
  - `15min-lead/wind/references/papers.md`
  - `15min-lead/wind/src/xinyang_wind15/remote_results.py`
  - `15min-lead/wind/tests/test_model_shapes.py`
  - `15min-lead/wind/tests/test_remote_results.py`

## 为什么这样做

目标不是立即把液态神经网络推成主线，而是用最低成本回答一个更关键的问题：

- 在单机纯时序控制实验里，`CfC` 相比现有 `GRU/TCN` 是否有可观增益？

如果单机层面都没有明显优势，就不值得直接往“图编码器 + CfC”复杂混合结构继续投入。

实现上优先复用了官方成熟实现，而不是手写论文版本：

- 来源：
  - `ncps` 官方仓库：<https://github.com/mlech26l/ncps>
  - 官方 `PyTorch CfC` 实现：<https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc.py>
- 采用原因：
  - 作者维护
  - API 清晰
  - 能直接嵌入现有 `PyTorch` 训练流程
- 本地适配：
  - 封装成与现有 `GRU/TCN` 一致的 `[batch, steps, turbines, features] -> [batch, turbines]` 接口
- 许可证/使用限制检查：
  - 使用公开官方实现，无额外私有依赖

## 做了哪些验证

### 1. 单元测试

运行：

```bash
pytest 15min-lead/wind/tests/test_model_shapes.py 15min-lead/wind/tests/test_remote_results.py
```

结果：

- `10` 项测试全部通过

### 2. 本地最小 smoke 训练

运行：

```bash
python 15min-lead/wind/scripts/train_cfc_baseline.py --turbine-id S29 --feature-columns ws_mean --max-turbines 1 --tail-timestamps 12000 --lookback-steps 16 --batch-size 128 --epochs 1 --hidden-size 32 --backbone-units 32 --output-dir 15min-lead/wind/artifacts/local_debug/cfc_smoke_s29_ws_only
```

结果：

- 训练脚本可启动
- 参数解析正确
- 成功输出：
  - `metrics.csv/json`
  - `summary.json`
  - `training_history.csv`
  - `val/test_per_turbine.csv`
  - `val/test_predictions.csv`
  - `cfc_baseline.pt`

本次 smoke 的 `S29 + ws_mean only` 测试集结果约为：

- `RMSE = 1.5495`
- `MAE = 1.0343`

这里只用于验证流程，不用于判断模型优劣。

## 当前风险、阻塞和下一步建议

- 风险：
  - 当前只验证了“脚本和流程能跑通”，还没有和既有 `GRU/TCN` 在同一单机设置下做严格对照。
  - `CfC` 的连续时间优势在规则 `15min` 采样场景里未必能转化成明显收益。
- 阻塞：
  - 服务器环境若尚未安装 `ncps`，提交前需要先补依赖。
- 下一步建议：
  - 先做两组单机对照：
    - `S29 + ws_mean only`
    - `S29 + ws_mean + wd_mean + nacelle_mean + ws_std`
  - 对照对象保持为：
    - `GRU`
    - `TCN`
    - `CfC`
  - 若 `CfC` 在单机上没有稳定收益，就不要继续扩展到图混合结构。
  - 若 `CfC` 有明确收益，再考虑：
    - `mixed_memory`
    - `pure / no_gate` 模式
    - 更长 lookback

## 涉及的数据、服务器或 GitHub 操作

- 本地使用了现有新洋数据做 smoke 训练，未新增数据副本。
- 本地安装了额外依赖：
  - `python -m pip install ncps`
- 未执行服务器操作。
- 未执行 Git 提交或推送。

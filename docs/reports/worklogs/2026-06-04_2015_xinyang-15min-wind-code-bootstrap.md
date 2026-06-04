# 2026-06-04 新洋 15 分钟风速代码启动报告

## 本次目标

- 在 `15min-lead/wind/` 下启动首批可执行代码实现。
- 优先使用成熟上游库与官方实现思路，先完成本地可运行的 baseline 和深度学习入口。
- 完成本地烟雾测试，确认脚本不是“只写不跑”。

## 本次完成内容

### 1. 新建实验本地代码包

新增目录：

- `15min-lead/wind/src/xinyang_wind15/`
- `15min-lead/wind/tests/`

新增核心模块：

- `schema.py`
- `settings.py`
- `splits.py`
- `loading.py`
- `features.py`
- `metrics.py`
- `baselines.py`
- `windows.py`
- `graph.py`
- `gru.py`

### 2. 新增依赖清单

新增：

- `15min-lead/wind/requirements.txt`

其中本次实际安装并验证了：

- `numpy`
- `pandas`
- `scikit-learn`
- `lightgbm`
- `statsmodels`
- `torch`
- `pyarrow`
- `openpyxl`

### 3. 新增可执行脚本

- `15min-lead/wind/scripts/run_local_baselines.py`
  - 支持：
    - persistence
    - seasonal persistence
    - LightGBM
    - farm-mean ARIMA
- `15min-lead/wind/scripts/build_window_dataset.py`
  - 构建未来 GRU / 图时空模型复用的窗口张量
- `15min-lead/wind/scripts/train_gru_baseline.py`
  - 训练一个本地多机组 GRU 基线

### 4. 新增测试

- `15min-lead/wind/tests/test_pipeline_utils.py`

已通过：

- `pytest 15min-lead\\wind\\tests -q`

### 5. 新增实现来源说明

- `15min-lead/wind/references/implementation_sources.md`

明确记录了本次优先参考的成熟来源：

- LightGBM 官方仓库与文档
- statsmodels 官方仓库与文档
- PyTorch 官方 GRU 文档
- 后续拟参考的 Graph WaveNet 与 TCN GitHub 仓库

## 为什么这样做

- 先把数据读取、切分、特征、评估和窗口化这套“公共地基”搭好，后面的 LightGBM、GRU、TCN、图时空模型都能复用，避免重复造轮子。
- baseline 先本地跑通，能尽早暴露数据切分、缺失值和指标口径问题，比直接上复杂深度模型更稳妥。
- GRU 先作为第一个可运行的深度学习基线，是为了把“深度学习主线”尽快从方案推进到代码。

## 本次实际验证

### 1. 单元测试

- `pytest 15min-lead\\wind\\tests -q`
- 结果：`2 passed`

### 2. 编译检查

- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts`
- 结果：通过

### 3. baseline 烟雾测试

命令：

- `python 15min-lead\\wind\\scripts\\run_local_baselines.py --max-turbines 4 --tail-timestamps 12000 --skip-sarima --output-dir 15min-lead\\wind\\artifacts\\local_debug\\baseline_smoke`

结果：

- persistence、seasonal persistence、LightGBM 均成功运行
- 输出位于：
  - `15min-lead/wind/artifacts/local_debug/baseline_smoke/`

其中：

- persistence 在该调试子集上的 test `RMSE` 约为 `0.5649`
- LightGBM 在该调试子集上的 test `RMSE` 约为 `0.8121`

### 4. ARIMA 烟雾测试

命令：

- `python 15min-lead\\wind\\scripts\\run_local_baselines.py --max-turbines 4 --tail-timestamps 12000 --sarima-train-tail-points 1000 --sarima-max-eval-points 128 --output-dir 15min-lead\\wind\\artifacts\\local_debug\\baseline_smoke_with_arima`

结果：

- farm-mean `ARIMA` 成功运行
- 选择到的参数是 `order = [1, 0, 0]`
- 输出位于：
  - `15min-lead/wind/artifacts/local_debug/baseline_smoke_with_arima/`

说明：

- 为保证本地调试可完成，本次对 ARIMA 使用了较短训练尾窗和较短验证/测试长度
- 后续要跑更完整的对照时，可再放宽该限制

### 5. 窗口数据集烟雾测试

命令：

- `python 15min-lead\\wind\\scripts\\build_window_dataset.py --max-turbines 4 --tail-timestamps 12000 --lookback-steps 32 --output-dir 15min-lead\\wind\\artifacts\\local_debug\\window_smoke`

结果：

- 成功生成窗口张量
- 样本形状：
  - `x_shape = [3844, 32, 4, 5]`
  - `y_shape = [3844, 4]`

### 6. GRU 烟雾测试

命令：

- `python 15min-lead\\wind\\scripts\\train_gru_baseline.py --epochs 3 --batch-size 128 --max-turbines 4 --tail-timestamps 12000 --output-dir 15min-lead\\wind\\artifacts\\local_debug\\gru_smoke`

结果：

- GRU 成功训练并输出结果
- 输出位于：
  - `15min-lead/wind/artifacts/local_debug/gru_smoke/`
- 调试子集指标：
  - val `RMSE` 约 `1.1181`
  - test `RMSE` 约 `0.8767`

说明：

- 当前 GRU 只是一个 smoke 级深度学习基线，尚未做充分调参
- 在该调试子集上，效果暂时弱于 persistence，这属于预期现象，不代表最终深度学习路线无效

## 本次踩到并已处理的问题

- LightGBM 在只含测试时段的极短尾窗下会因为缺少 train/val 数据而失败。
  - 已增加清晰报错说明。
- farm-mean ARIMA 初版在 `statsmodels` 的滚动 `append()` 上因列结构不匹配失败。
  - 已修正为与原始 `endog` 对齐的 DataFrame 追加方式。
- farm-mean 序列补齐为严格 `15min` 频率后会出现少量缺口。
  - 已在建序列时加入保守的 `ffill().bfill()`。
- ARIMA 全量滚动验证在本地调试中耗时偏高。
  - 已加入：
    - `--sarima-train-tail-points`
    - `--sarima-max-eval-points`

## 当前风险

- 目前 baseline 和 GRU 的实际 smoke test 仅在 `4` 台机组、`12000` 个时间戳子集上验证，不代表全量 46 台最终效果。
- ARIMA 当前以本地友好的短尾窗配置为主，完整全量对照还需要进一步跑长实验。
- 还没有实现：
  - TCN
  - 图时空主模型
  - 1 分钟聚合特征
  - 观测塔宽表 ablation

## 下一步建议

- 先把 baseline 和 GRU 扩展到全量 `46` 台机组。
- 再加入：
  - 观测塔特征
  - 1 分钟聚合特征
- 然后开始实现：
  - TCN
  - GraphWaveNet 风格图时空主模型


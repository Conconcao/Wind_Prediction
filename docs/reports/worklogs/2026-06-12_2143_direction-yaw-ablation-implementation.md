# 2026-06-12 Direction Yaw Ablation Implementation

## 本次目标

- 把新洋 `15min` 风向/机组朝向消融实验从想法落成可跑代码。
- 保持现有 `GWNet` 训练主链不分叉，优先复用已有 store 和训练流程。
- 为服务器端补齐最短路径的 `bsub` 入口。

## 实际修改了什么

- 在 `15min-lead/wind/src/xinyang_wind15/feature_presets.py` 新增了方向类特征预设：
  - `direction_wd_only`
  - `direction_wd_yaw`
  - `direction_wd_yaw_error`
- 在 `15min-lead/wind/src/xinyang_wind15/features.py` 新增了偏航失配派生特征：
  - `yaw_error_deg`
  - `yaw_error_sin`
  - `yaw_error_cos`
  - `yaw_error_abs`
- 在 `15min-lead/wind/scripts/build_window_store.py` 与
  `15min-lead/wind/src/xinyang_wind15/graph.py` 中补充了方向图所需的几何信息：
  - `bearing_matrix.npy`
  - bearing 计算函数
- 在 `15min-lead/wind/src/xinyang_wind15/gwnet.py` 中扩展了 `GraphWaveNetLite`：
  - 支持批级 `batched support`
  - 支持额外动态 support 通道
- 在 `15min-lead/wind/scripts/train_gwnet_from_store.py` 中新增了动态方向 support 逻辑：
  - `--dynamic-directional-support`
  - `--direction-support-source`
  - `--direction-support-sigma-deg`
- 在 `15min-lead/wind/src/xinyang_wind15/sequence.py` 中扩展了评估入口，
  允许 `GWNet` 在验证/测试时使用同样的动态 support。
- 新增参数化 `LSF` 作业脚本：
  - `jobs/lsf/xinyang_build_store_direction_ablation.lsf`
  - `jobs/lsf/xinyang_train_gwnet_direction_ablation.lsf`
- 更新了 `15min-lead/wind/README.md`，补充 `D0-D4` 方向/朝向实验说明。
- 新增并更新了相关测试。

## 为什么这样做

- 最短路径不是再造一套新模型，而是在现有 `GWNet` 管线里只改“特征”和“support”。
- 这样可以把实验差异尽量压缩到：
  - 是否加入方向/朝向节点特征
  - 是否把方向信息用于空间依赖重排
- 参数化 `LSF` 脚本比复制多份近似脚本更短、更稳，也更不容易提交错作业。

## 做了哪些验证

- 单元测试：
  - `pytest 15min-lead/wind/tests/test_feature_presets.py 15min-lead/wind/tests/test_pipeline_utils.py 15min-lead/wind/tests/test_window_store.py 15min-lead/wind/tests/test_model_shapes.py`
  - 结果：`20 passed`
- 本地 store 烟测：
  - 命令：
    - `python 15min-lead/wind/scripts/build_window_store.py --feature-preset direction_wd_yaw_error --max-turbines 2 --output-dir 15min-lead/wind/artifacts/local_debug/direction_store_smoke_full`
  - 结果：
    - `33095` timestamps
    - `30506` valid windows
    - `20981 / 6392 / 3133` train/val/test windows
    - `8` 个方向/偏航相关特征成功写入 store
- 本地训练烟测：
  - 命令：
    - `python 15min-lead/wind/scripts/train_gwnet_from_store.py --store-dir 15min-lead/wind/artifacts/local_debug/direction_store_smoke_full --support-mode distance_correlation --epochs 1 --batch-size 32 --num-workers 0 --dynamic-directional-support --direction-support-source wd_sincos --output-dir 15min-lead/wind/artifacts/local_debug/gwnet_direction_dyn_smoke`
  - 结果：
    - 训练成功完成
    - summary 中显示 `dynamic_directional_support = true`
    - `supports = 4`
    - 本地 `cpu` 上可正常完成一个 epoch

## 当前风险、阻塞和下一步建议

- 风险：
  - 当前 `D4` 是“方向驱动的动态 support 最小实现”，仍然是轻量近似，不是完整物理尾流模型。
  - 动态 support 目前依据最近一步的每机组风向，不含更复杂的时滞或尾流传播时间。
- 阻塞：
  - 无直接阻塞，服务器端已具备提交条件。
- 下一步建议：
  - 先按同一超参跑完：
    - `D1`: `direction_wd_only`
    - `D2`: `direction_wd_yaw`
    - `D3`: `direction_wd_yaw_error`
    - `D4`: `D3 + dynamic directional support`
  - 比较顺序建议：
    - 先 `D1`
    - 再 `D3`
    - 最后 `D4`
  - 如果 `D4` 明显优于 `D3`，再考虑更重的方向感知图结构或上游聚合。

## 涉及的数据、服务器或 GitHub 操作

- 本轮未改动原始数据。
- 本轮新增了服务器端 `LSF` 作业入口，但尚未提交服务器作业。
- 本轮未执行 Git 提交或推送。
- 本地调试产物仅落在：
  - `15min-lead/wind/artifacts/local_debug/direction_store_smoke_full`
  - `15min-lead/wind/artifacts/local_debug/gwnet_direction_dyn_smoke`
  且未进入当前 Git 状态。

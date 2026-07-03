# 本次目标

- 将淮安场站 `15min-lead` 风速预测任务改成一条独立的新流水线：
  - 只使用 `1min` 分辨率原始 SCADA；
  - 按时间顺序 `7:1:2` 划分；
  - 分别支持 `LSTM / GRU / TFT / GNN`。
- 让这条新流水线可以在本地最小化验证，并能在服务器端直接提交作业。

# 实际修改了什么

- 新增 `1min-only` 数据与切分辅助模块：
  - `15min-lead/wind/src/xinyang_wind15/raw_one_min.py`
- 新增两类模型实现：
  - `15min-lead/wind/src/xinyang_wind15/lstm.py`
  - `15min-lead/wind/src/xinyang_wind15/tft.py`
- 新增 `1min-only` store 构建脚本：
  - `15min-lead/wind/scripts/build_raw_one_min_store.py`
- 新增统一的序列模型训练入口：
  - `15min-lead/wind/scripts/train_seq_model_from_store.py`
  - 通过 `--arch {lstm,gru,tft}` 切换
- 新增淮安 `1min raw` 配置：
  - `15min-lead/wind/configs/splits/huaian_1min_raw_7_1_2.yaml`
  - `15min-lead/wind/configs/splits/huaian_1min_raw_7_1_2_server.yaml`
- 新增服务器 `LSF` 脚本：
  - `jobs/lsf/huaian_build_store_1min_raw_ws.lsf`
  - `jobs/lsf/huaian_train_lstm_1min_raw_ws.lsf`
  - `jobs/lsf/huaian_train_gru_1min_raw_ws.lsf`
  - `jobs/lsf/huaian_train_tft_1min_raw_ws.lsf`
  - `jobs/lsf/huaian_train_gwnet_1min_raw_ws.lsf`
- 新增测试：
  - `15min-lead/wind/tests/test_raw_one_min.py`

# 为什么这样做

- 旧淮安流程本质上是：
  - `15min` 聚合主表作为锚点；
  - `1min` 仅作为附加统计特征。
- 这与“只用 `1min` 原始 SCADA 做风速预测”的目标不一致，所以不能在原有 `15min` 主表路径上继续修补。
- 这次改造直接把：
  - 标签分辨率定义为 `1min`；
  - 预测时距定义为 `15` 个 `1min` step；
  - 图模型和序列模型统一挂到同一个磁盘版 `window store` 上。
- 这样做的好处是：
  - 数据定义更干净；
  - 4 个模型共享同一份样本切分；
  - 服务器端只需要建一次 store，后续多模型复用。

# 关键数据判断

- 本地检查淮安 `1min` 原始文件后确认：
  - `ws` 基本完整；
  - `wd` 全空；
  - `nacelle_angle` 全空；
  - `power` 虽完整，但属于机组功率状态，不纳入本轮“只用原始风速链路”的主特征。
- 因此当前这版 `1min-only` 实验，实际可稳定使用的原始特征只有：
  - `ws`
- 也就是说，当前落地的是：
  - “全场多机组联合建模”
  - “输入为过去 `60` 分钟各机组原始风速序列”
  - “目标为 `t+15min` 的风速点预测”

# 做了哪些验证

- 单元测试：
  - `python -m pytest 15min-lead/wind/tests/test_raw_one_min.py`
  - 结果：`3 passed`
- `LSF` 脚本语法检查：
  - 对新增 5 个 `jobs/lsf/*.lsf` 运行 `bash -n`
  - 结果通过
- 本地 smoke store：
  - `python 15min-lead/wind/scripts/build_raw_one_min_store.py --config 15min-lead/wind/configs/splits/huaian_1min_raw_7_1_2.yaml --max-turbines 2 --tail-timestamps 1200 --output-dir 15min-lead/wind/artifacts/local_debug/huaian_1min_raw_store_smoke`
  - 成功生成 `2` 台机组、`1200` 个时间点、`1126` 个有效窗口
- 本地 smoke 训练：
  - `LSTM`：`huaian_1min_lstm_smoke`
  - `GRU`：`huaian_1min_gru_smoke`
  - `TFT`：`huaian_1min_tft_smoke_rerun`
  - `GWNet`：`huaian_1min_gwnet_smoke`
  - 4 条训练入口均已跑通并成功输出：
    - `metrics.json`
    - `summary.json`
    - `val_predictions.csv`
    - `test_predictions.csv`

# 当前风险、阻塞和下一步建议

- 当前风险：
  - 用户口中的 “15min-lead” 若想定义成“未来 15 分钟平均风速”而不是 “`t+15min` 点预测”，则需要再改目标构造逻辑。
  - 淮安 `1min` 原始文件中没有可用风向，因此这版并不是“风速+风向”输入，而是“纯风速历史”输入。
  - `TFT` 当前是一个紧凑版 `TFT-style` 实现，不是完整工业级 `Temporal Fusion Transformer` 全量复刻。
- 下一步建议：
  - 先按当前定义在服务器端跑完整 4 模型，确认全场 `1min raw ws` 这条最干净基线的上限。
  - 若后续补到可用的 `1min` 风向，再把 `wd_sin/wd_cos` 作为第二版增强输入。
  - 若你希望目标改成“未来 15 分钟平均风速”，下一轮我直接把 target 构造切过去，不需要重写训练器。

# 涉及的数据、服务器或 GitHub 操作

- 本地使用的数据文件：
  - `C:\Users\caosh\Desktop\WTPC\data\huaian\ALL_TURBINES_1min_202307-202512.parquet`
  - `C:\Users\caosh\Desktop\WTPC\data\huaian\风机基本信息.csv`
- 服务器配置要求的数据文件：
  - `/home3/s502024280003/Wind_Prediction/data/huaian/ALL_TURBINES_1min_202307-202512.parquet`
  - `/home3/s502024280003/Wind_Prediction/data/huaian/风机基本信息.csv`
- 本轮尚未自动提交服务器作业；新增脚本已准备好，待代码推送后可直接 `bsub`。

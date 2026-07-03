# 本次目标

- 修复淮安场站因缺少 `风机基本信息.csv` 导致的元数据读取问题，改为支持 `风机基本信息汇总.csv`。
- 在淮安 `15min-lead`、`1min` 原始 SCADA 风速预测链路中新增 `CNN+LSTM` 深度学习模型。

# 实际修改了什么

- 修改 `15min-lead/wind/src/xinyang_wind15/loading.py`
  - `turbine_meta` 回退匹配优先支持 `*风机基本信息汇总*.csv`。
  - `load_turbine_metadata()` 新增 `site` 参数，可按场站筛选汇总元数据。
  - 加入 `turbine_id` 去重检查，避免跨场站汇总表误混入重复机组。
- 修改以下脚本的元数据加载调用，统一传入 `site=settings.site`：
  - `15min-lead/wind/scripts/build_raw_one_min_store.py`
  - `15min-lead/wind/scripts/build_window_store.py`
  - `15min-lead/wind/scripts/build_window_dataset.py`
- 修改淮安配置文件中的元数据路径：
  - `15min-lead/wind/configs/splits/huaian_1min_raw_7_1_2.yaml`
  - `15min-lead/wind/configs/splits/huaian_1min_raw_7_1_2_server.yaml`
  - `15min-lead/wind/configs/splits/huaian_7_2_1.yaml`
  - `15min-lead/wind/configs/splits/huaian_7_2_1_server.yaml`
- 新增 `15min-lead/wind/src/xinyang_wind15/cnn_lstm.py`
  - 实现多机组联合建模的 `CNN+LSTM`。
- 修改 `15min-lead/wind/scripts/train_seq_model_from_store.py`
  - 新增 `cnn_lstm` 架构选项。
  - 新增 `--conv-channels`、`--conv-kernel-size` 参数。
  - 在 `summary.json` 中写入卷积超参数。
- 新增服务器脚本 `jobs/lsf/huaian_train_cnn_lstm_1min_raw_ws.lsf`。
- 修改测试 `15min-lead/wind/tests/test_raw_one_min.py`
  - 新增“汇总元数据按 site 过滤”的单元测试。

# 为什么这样做

- 根因不是“文件找不到”本身，而是淮安现在只有跨场站汇总版机组元数据。如果仍按单场站文件名硬编码，服务器侧会持续报错。
- 只改文件名不够。汇总表若不按 `site` 过滤，会把其它场站机组坐标也读进来，后续邻接矩阵、距离矩阵和图模型输入都会错。
- `CNN+LSTM` 适合当前这类“多机组联合 + 短历史窗口 + 单步超短期外推”场景，可作为比纯 `LSTM/GRU` 更强一点的时序卷积基线。

# 做了哪些验证

- `python -m pytest 15min-lead/wind/tests/test_raw_one_min.py`
  - 结果：`4 passed`
- 本地构建淮安 1min store smoke：
  - `python 15min-lead/wind/scripts/build_raw_one_min_store.py --config 15min-lead/wind/configs/splits/huaian_1min_raw_7_1_2.yaml --max-turbines 2 --tail-timestamps 1200 --output-dir 15min-lead/wind/artifacts/local_debug/huaian_1min_raw_store_smoke_meta_fix`
  - 结果：成功生成 store，说明 `D:/Power_prediction/Data/风机基本信息汇总.csv` 可被正确读取并按淮安场站过滤。
- 本地训练 `CNN+LSTM` smoke：
  - `python 15min-lead/wind/scripts/train_seq_model_from_store.py --store-dir 15min-lead/wind/artifacts/local_debug/huaian_1min_raw_store_smoke_meta_fix --arch cnn_lstm --epochs 1 --batch-size 64 --hidden-size 16 --num-layers 1 --conv-channels 16 --conv-kernel-size 5 --dropout 0.1 --output-dir 15min-lead/wind/artifacts/local_debug/huaian_1min_cnn_lstm_smoke_meta_fix`
  - 结果：成功完成训练与评估输出。

# 当前风险、阻塞和下一步建议

- 当前淮安 `1min` 原始 SCADA 样本中，`wd` 与 `nacelle_angle` 先前抽样检查几乎为空；即使代码支持方向链路，是否真正能用于淮安训练仍要以完整数据非空率复核。
- 服务器端需要确保汇总元数据文件实际放在：
  - `/home3/s502024280003/Wind_Prediction/data/huaian/风机基本信息汇总.csv`
- 下一步建议：
  - 先在服务器重新构建淮安 `1min` store。
  - 并行训练 `LSTM / GRU / TFT / GNN / CNN+LSTM`。
  - 训练完成后统一回传结果，再比较空间联合建模与纯时序模型差异。

# 涉及的数据、服务器或 GitHub 操作

- 本地读取的数据与元数据：
  - `C:\Users\caosh\Desktop\WTPC\data\huaian\ALL_TURBINES_1min_202307-202512.parquet`
  - `D:\Power_prediction\Data\风机基本信息汇总.csv`
- 本次未执行 Git 提交或推送。
- 服务器侧尚未执行；待用户拉取最新代码后重跑淮安 store 和训练脚本。

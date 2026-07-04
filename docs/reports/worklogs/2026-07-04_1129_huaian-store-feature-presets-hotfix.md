# 本次目标

- 修复服务器端 `huaian_build_store_hubws_from_1minagg.lsf` 运行时报出的
  `cannot import name 'validate_feature_columns'` 错误。

# 实际修改了什么

- 将 `15min-lead/wind/src/xinyang_wind15/feature_presets.py` 中与当前
  `build_window_store.py` 兼容所需的缺失内容补回已提交版本：
  - `validate_feature_columns`
  - `multivariate_directional`
  - `huaian_directional_core`

# 为什么这样做

- 当前 `main` 上再次出现了接口不一致：
  - `build_window_store.py` 已经导入 `validate_feature_columns`
  - 但远端 `feature_presets.py` 里还没有这个函数
- 这不是当前作业参数问题，而是代码版本不一致导致的直接导入失败。
- 最短修复路径是把 `feature_presets.py` 补齐到与 `build_window_store.py` 相同接口层级。

# 做了哪些验证

- 核对 `HEAD` 后确认：
  - `build_window_store.py` 确实导入了 `validate_feature_columns`
  - 远端已提交版本的 `feature_presets.py` 确实缺这个函数
- 本地核对新增 preset 与校验函数实现后，确认其职责仅为：
  - 返回淮安方向预设
  - 检查所选特征是否存在且不是整列全空

# 当前风险、阻塞和下一步建议

- 这次仍属于兼容性热修。
- 服务器端不需要重跑聚合作业；如果聚合文件已经成功生成，只需要：
  1. `git pull --ff-only origin main`
  2. 重提 `huaian_build_store_hubws_from_1minagg.lsf`
  3. 成功后再提 `huaian_train_gwnet_hubws_from_1minagg.lsf`

# 涉及的数据、服务器或 GitHub 操作

- 本次未新增数据文件。
- 本次需要再次推送 GitHub，并由服务器重新拉取代码。

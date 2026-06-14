# 本次目标

修复单机脚本在服务器端手工运行时默认读取本地 `Windows` split 配置、导致数据路径报错的问题。

## 实际修改了什么

- 在 `15min-lead/wind/src/xinyang_wind15/settings.py` 新增环境感知的默认配置解析函数：
  - `default_split_config_path(...)`
- 更新以下脚本的 `--split-config` 默认值，不再硬编码本地版配置：
  - `15min-lead/wind/scripts/train_cfc_baseline.py`
  - `15min-lead/wind/scripts/train_gru_baseline.py`
  - `15min-lead/wind/scripts/train_tcn_baseline.py`
  - `15min-lead/wind/scripts/build_window_dataset.py`
  - `15min-lead/wind/scripts/build_window_store.py`
  - `15min-lead/wind/scripts/run_local_baselines.py`
- 新增测试：
  - `15min-lead/wind/tests/test_settings.py`

## 为什么这样做

根因不是 `CfC` 模型本身，而是多份脚本都把默认 `--split-config` 写死成了本地版：

- `xinyang_7_2_1.yaml`

该文件里的数据路径是：

- `C:/Users/caosh/Desktop/WTPC/...`

所以在服务器上手工运行、不显式传 `--split-config` 时，脚本必然去找 `Windows` 路径并报错。

这不是单个命令写错，而是默认行为设计不对。  
因此修复应落在“默认配置选择逻辑”上，而不是只靠口头提醒每次都加参数。

## 做了哪些验证

运行：

```bash
pytest 15min-lead/wind/tests/test_settings.py 15min-lead/wind/tests/test_model_shapes.py 15min-lead/wind/tests/test_remote_results.py
```

结果：

- `12` 项测试全部通过

验证点包括：

- Windows / 本地分支优先本地版 split 配置
- Linux / 服务器分支优先服务器版 split 配置
- 既有模型形状测试与远程打包测试未被破坏

## 当前风险、阻塞和下一步建议

- 风险：
  - 这次修复只解决“默认 split 配置”问题，不改变服务器端是否已经装好 `ncps` 依赖。
- 阻塞：
  - 若服务器环境尚未 `pip install ncps` 或未重装 requirements，`CfC` 仍会因缺依赖报错。
- 下一步建议：
  - 服务器端先用显式 `--split-config` 立刻继续当前任务，避免等待。
  - 拉取本次修复后，后续手工运行这些脚本时就不必再额外记住服务器版配置路径。

## 涉及的数据、服务器或 GitHub 操作

- 本次修复不涉及原始数据变更。
- 需要 Git 提交并推送，供服务器端拉取。

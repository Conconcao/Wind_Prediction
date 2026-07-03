# 本次目标

- 修复淮安 `1min` store 任务在服务器端因 `schema.py` 常量缺失导致的启动失败。

# 实际修改了什么

- 核对 `logs/6303.huaian_1min_store.err` 后，确认报错来自：
  - `cannot import name 'RAW_SCADA_15MIN_DIRECTION_TO_CANONICAL' from 'xinyang_wind15.schema'`
- 对比已推送版本与本地工作区，确认上一次提交遗漏了 `15min-lead/wind/src/xinyang_wind15/schema.py` 中的新常量定义。
- 本次准备补推：
  - `RAW_SCADA_15MIN_DIRECTION_TO_CANONICAL`
  - `SCADA_15MIN_DIRECTION_COLUMNS`
  - 15 分钟 SCADA 的若干兼容列映射

# 为什么这样做

- 根因不是服务器环境，也不是数据路径。
- `loading.py` 已经依赖新的方向字段常量，但仓库上次推送的 `schema.py` 仍是旧版本，导致 Python 在导入阶段直接退出，作业还没进入真正的数据读取逻辑。
- 这种问题必须修依赖闭环，而不是继续改提交脚本或重试作业。

# 做了哪些验证

- 本地检查：
  - `git diff -- 15min-lead/wind/src/xinyang_wind15/schema.py`
  - 已确认差异正是缺失的常量定义。
- 语法检查：
  - `python -m compileall 15min-lead/wind/src/xinyang_wind15/schema.py 15min-lead/wind/src/xinyang_wind15/loading.py`
- 单元测试：
  - `python -m pytest 15min-lead/wind/tests/test_raw_one_min.py`
  - 结果：`4 passed`

# 当前风险、阻塞和下一步建议

- 当前阻塞点仅剩服务器端拉取最新提交并重新运行 `store`。
- 若拉取后仍报错，再看新的 `.err`，但大概率这次会越过 import 阶段。
- 若新的错误进入数据读取阶段，再继续排数据文件或元数据内容问题。

# 涉及的数据、服务器或 GitHub 操作

- 本次未改数据文件。
- 将补充一次 Git 提交并推送到 `origin/main`，供服务器端 `git pull` 后重跑。

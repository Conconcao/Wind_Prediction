# 本次目标

- 按用户要求，回退刚刚加入仓库的淮安 `LSF` 环境激活加固改动。
- 保留前一轮已经可用的淮安实验脚本、功率评估逻辑和结果分析，不扩大回退范围。

# 实际修改了什么

- 回退了提交 `2c85e43` 中的内容：
  - 删除 `docs/reports/worklogs/2026-07-03_1558_huaian-lsf-env-hardening.md`
  - 恢复以下 8 个脚本到加固前状态：
    - `jobs/lsf/huaian_build_store_hubws_joint.lsf`
    - `jobs/lsf/huaian_train_gwnet_hubws_joint.lsf`
    - `jobs/lsf/huaian_build_store_directional_joint.lsf`
    - `jobs/lsf/huaian_train_gwnet_directional_joint.lsf`
    - `jobs/lsf/huaian_build_store_directional_derived_core.lsf`
    - `jobs/lsf/huaian_train_gwnet_directional_derived_core.lsf`
    - `jobs/lsf/huaian_build_store_directional_derived_ctx.lsf`
    - `jobs/lsf/huaian_train_gwnet_directional_derived_ctx.lsf`

# 为什么这样做

- 用户明确表示环境将由自己手动激活，因此不希望作业脚本继续内置环境探测和激活逻辑。
- 这意味着脚本应恢复为更简洁的版本，只保留实验本身的执行命令，不替用户决定环境初始化方式。
- 这次回退只针对最后一轮的环境层改动，不回退淮安 `15min-lead` 的实验配置、功率评估口径和 `GWNet` 作业脚本主体，避免把已确认有价值的工作一并撤掉。

# 做了哪些验证

- 使用 `git revert --no-commit 2c85e43` 执行精确回退，确认回退范围仅覆盖最后一个提交涉及的 9 个文件。
- 未对其他已有未提交改动做任何回滚或清理。

# 当前风险、阻塞和下一步建议

- 当前风险：
  - 作业脚本恢复后，将再次依赖用户在提交前手动激活服务器环境。
  - 如果提交系统不继承当前 shell 的环境，仍可能出现 `python/conda` 不可见问题。
- 下一步建议：
  - 在服务器端手动激活环境后，再提交作业。
  - 若仍报错，优先检查提交脚本执行时是否真的继承了当前环境，而不是继续在脚本内部做隐式修复。

# 涉及的数据、服务器或 GitHub 操作

- 本轮仅涉及 Git 回退，不涉及数据改动。
- 计划将回退结果提交并推送到：
  - `git@github.com:Conconcao/Wind_Prediction.git`

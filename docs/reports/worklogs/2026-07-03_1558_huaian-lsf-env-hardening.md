# 本次目标

- 排查淮安场站首个 `store` 作业 `6295` 在服务器端直接 `EXIT 127` 的问题。
- 修复淮安新增 `LSF/bsub` 脚本的批处理环境激活逻辑，保证计算节点上可以稳定找到 `python/conda`。

# 实际修改了什么

- 更新了以下 8 个淮安 `LSF` 脚本：
  - `jobs/lsf/huaian_build_store_hubws_joint.lsf`
  - `jobs/lsf/huaian_train_gwnet_hubws_joint.lsf`
  - `jobs/lsf/huaian_build_store_directional_joint.lsf`
  - `jobs/lsf/huaian_train_gwnet_directional_joint.lsf`
  - `jobs/lsf/huaian_build_store_directional_derived_core.lsf`
  - `jobs/lsf/huaian_train_gwnet_directional_derived_core.lsf`
  - `jobs/lsf/huaian_build_store_directional_derived_ctx.lsf`
  - `jobs/lsf/huaian_train_gwnet_directional_derived_ctx.lsf`
- 每个脚本都新增了：
  - `#BSUB -L /bin/bash`
  - `CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"`
  - 优先显式 `source "$HOME/miniconda3/etc/profile.d/conda.sh"` 再 `conda activate`
  - 若 `conda` 不在 PATH，则回退到 `"$CONDA_ROOT/envs/$ENV_NAME/bin/python"`
  - 若激活后仍找不到 `python`，明确 `exit 127` 并打印错误信息

# 为什么这样做

- `bjobs -l 6295` 显示：
  - 作业几乎立即退出；
  - `CPU time` 和 `MEM` 基本为 `0`；
  - 退出码是 `127`
- 这更像 shell 级别的 “command not found”，而不是 Python 脚本内部的数据或参数错误。
- 在超算登录节点上能 `conda activate`，不代表批处理计算节点的非交互 shell 也自动带有同样的 `PATH` 和 `conda` 初始化。
- 因此应直接从已知安装路径 `$HOME/miniconda3` 加载 `conda.sh`，而不是依赖批处理环境碰巧继承交互式 shell 状态。

# 做了哪些验证

- 对以上 8 个 `LSF` 脚本运行了 `bash -n` 语法检查。
- 检查结果通过。
- 本地 `WSL` 在语法检查时打印了若干 `localhost/NAT` 警告，但未影响 `bash -n` 成功返回。

# 当前风险、阻塞和下一步建议

- 当前风险：
  - 虽然 `127` 的最可能根因已经覆盖，但服务器端旧作业 `6295` 本身不会自动恢复，需要重新拉代码并重提。
  - 若服务器上的 `conda` 实际安装路径不是 `$HOME/miniconda3`，需要在提交前临时指定：
    - `export CONDA_ROOT=/实际/miniconda3/路径`
- 下一步建议：
  - 先在服务器端 `git pull` 到最新提交后，仅重跑 `hubws store` 一次验证环境修复是否生效。
  - 若 `hubws store` 成功，再并行提交其他 `store`。
  - 如果仍失败，立即查看：
    - `tail -n 80 logs/<jobid>.huaian_store_hubws.out`
    - `tail -n 80 logs/<jobid>.huaian_store_hubws.err`
  - 这时日志里会直接暴露是：
    - `python not found`
    - `conda.sh` 路径不对
    - 还是进入了 Python 后的别的错误

# 涉及的数据、服务器或 GitHub 操作

- 本轮仅修改脚本与工作报告，不涉及原始数据变更。
- 目标服务器路径仍为：
  - `/home3/s502024280003/Wind_Prediction`
- 待完成操作：
  - 提交并推送上述 `LSF` 环境修复改动
  - 由用户在服务器端拉取后重新提交作业

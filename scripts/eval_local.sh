#!/usr/bin/env bash
# ====================================================================
# 本地完整评测脚本
# 对 12 标准配置 + 12 _random 配置（共 24 配置）按 EVAL_NUM 个 episode 跑评测
# 用于在正式提交前估算 Generalization 维度成绩
# 官方命令参考：https://xsparkai.com/goai-2026/
# ====================================================================
set -e

# ---------- 可配置变量（通过环境变量覆盖）----------
POLICY_DIR="${POLICY_DIR:-XPolicyLab/policy/YOUR_POLICY}"
POLICY_ENV="${POLICY_ENV:-YOUR_POLICY_CONDA_ENV}"
RUN_LABEL="${RUN_LABEL:-default}"
# 动作类型：ee（末端执行器）或 joint，须与最终提交一致
ACTION_TYPE="${ACTION_TYPE:-ee}"
# 每个配置评测的 episode 数（默认 20）
EVAL_NUM="${EVAL_NUM:-20}"

echo "=== GOAI 本地评测（Generalization 维度，24 配置 × ${EVAL_NUM} episode）==="
echo "POLICY_DIR  = ${POLICY_DIR}"
echo "POLICY_ENV  = ${POLICY_ENV}"
echo "RUN_LABEL   = ${RUN_LABEL}"
echo "ACTION_TYPE = ${ACTION_TYPE}"

# 调用 RoboDojo 官方脚本 robodojo.sh 的 eval 子命令
bash scripts/robodojo.sh eval \
  --dimension generalization \
  --policy-dir "${POLICY_DIR}" \
  --ckpt "${RUN_LABEL}" \
  --policy-env "${POLICY_ENV}" \
  --eval-env RoboDojo \
  --action-type "${ACTION_TYPE}" \
  --eval-num "${EVAL_NUM}"

echo "=== 本地评测完成，请查看 RoboDojo 输出的成功率汇总 ==="

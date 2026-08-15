#!/usr/bin/env bash
# ====================================================================
# GOAI 2026 冒烟测试脚本
# 仅评测 Generalization 维度，对 12 标准配置 + 12 _random 配置各运行 1 episode
# 目的：验证 Policy Server ↔ RoboDojo 链路是否打通，不代表正式成绩
# 官方命令参考：https://xsparkai.com/goai-2026/
# ====================================================================
set -e

# ---------- 可配置变量（通过环境变量覆盖）----------
# Policy 目录（XPolicyLab/policy/ 下或本项目 policy/ 下的子目录）
POLICY_DIR="${POLICY_DIR:-XPolicyLab/policy/YOUR_POLICY}"
# Policy 对应的 conda 环境名
POLICY_ENV="${POLICY_ENV:-YOUR_POLICY_CONDA_ENV}"
# 本次运行的 checkpoint 标签
RUN_LABEL="${RUN_LABEL:-default}"
# 动作类型：ee（末端执行器）或 joint，须与最终提交一致
ACTION_TYPE="${ACTION_TYPE:-ee}"

echo "=== GOAI 冒烟测试（Generalization 维度，24 配置）==="
echo "POLICY_DIR  = ${POLICY_DIR}"
echo "POLICY_ENV  = ${POLICY_ENV}"
echo "RUN_LABEL   = ${RUN_LABEL}"
echo "ACTION_TYPE = ${ACTION_TYPE}"

# 调用 RoboDojo 官方脚本 robodojo.sh 的 smoke 子命令
# --dimension generalization：仅评测泛化维度（12 标准 + 12 _random）
# --eval-num 1：每个配置只跑 1 episode（冒烟用）
bash scripts/robodojo.sh smoke \
  --dimension generalization \
  --policy-dir "${POLICY_DIR}" \
  --ckpt "${RUN_LABEL}" \
  --policy-env "${POLICY_ENV}" \
  --eval-env RoboDojo \
  --action-type "${ACTION_TYPE}" \
  --eval-num 1

echo "=== 冒烟测试完成（仅验证链路，不代表正式成绩）==="

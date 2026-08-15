#!/usr/bin/env bash
# ==========================================================================
# train_unified_act.sh — 训练赛事规则合规的"单一多任务 ACT checkpoint"
# --------------------------------------------------------------------------
# 背景：
#   赛事规则明确要求"审核和正式评测期间不更换模型/动作类型/协议行为"。
#   旧 MultiACT 方案在同一服务内按任务切换 ckpt，被官方认定为违规。
#   本脚本调用 XPolicyLab ACT 官方 cotrain recipe，产出 **唯一一个**
#   多任务联合 ckpt，配合 policy/UnifiedACT/ 包装器即可满足合规要求。
#
# 官方依据：
#   XPolicyLab/policy/ACT/README.md 明确写道：
#   "task_name is only the evaluation task; multi-task checkpoints can be
#    evaluated on different tasks without renaming the checkpoint
#    directory." —— 即单 ckpt 跨任务评测是官方支持的标准用法。
#
# 用法：
#   bash scripts/train_unified_act.sh [GPU_ID]
#     GPU_ID 默认 0；可用逗号分隔多 GPU（若上游 trainer 支持）。
#
# 产物路径示例：
#   XPolicyLab/policy/ACT/checkpoints/RoboDojo-cotrain-arx_x5-joint-0/
#
# 训练完成后：
#   将产物绝对路径填入 policy/UnifiedACT/deploy.yml::ckpt_dir
# ==========================================================================
set -e

GPU_ID="${1:-0}"
BENCH_NAME="RoboDojo"
CKPT_NAME="cotrain"            # 关键：cotrain = 多任务联合训练
ENV_CFG_TYPE="arx_x5"
ACTION_TYPE="joint"
SEED="0"

# --------------------------------------------------------------------------
# 0. 定位 XPolicyLab ACT 目录（支持常见路径覆盖）
# --------------------------------------------------------------------------
XPL_ROOT="${XPL_ROOT:-/mnt/workspace/RoboDojo/XPolicyLab}"
ACT_DIR="${ACT_DIR:-${XPL_ROOT}/policy/ACT}"

if [ ! -d "${ACT_DIR}" ]; then
    echo "[ERROR] XPolicyLab ACT 目录不存在: ${ACT_DIR}"
    echo "请先运行: bash scripts/setup_xpolicylab.sh"
    echo "或通过环境变量 XPL_ROOT / ACT_DIR 指定自定义路径。"
    exit 1
fi

echo "============================================================"
echo " UnifiedACT 单一多任务 ckpt 训练"
echo "   XPL_ROOT      = ${XPL_ROOT}"
echo "   ACT_DIR       = ${ACT_DIR}"
echo "   bench_name    = ${BENCH_NAME}"
echo "   ckpt_name     = ${CKPT_NAME}   (cotrain = 多任务联合)"
echo "   env_cfg_type  = ${ENV_CFG_TYPE}"
echo "   action_type   = ${ACTION_TYPE}"
echo "   seed          = ${SEED}"
echo "   gpu_id        = ${GPU_ID}"
echo "============================================================"

cd "${ACT_DIR}"

# --------------------------------------------------------------------------
# 1. 合并数据处理（若未处理过）
# --------------------------------------------------------------------------
# 官方提供合并数据则跳过；如需自行合并，使用 cotrain 作为 ckpt_name
# 在 data/RoboDojo/cotrain/ 下聚合所有 12 任务数据即可。
if [ ! -d "data/${BENCH_NAME}/${CKPT_NAME}" ]; then
    echo "[WARN] 未找到合并数据目录 data/${BENCH_NAME}/${CKPT_NAME}/"
    echo "       请按 XPolicyLab/policy/ACT/README.md 的 process_data.sh 流程"
    echo "       将 12 任务数据合并到 cotrain 目录后再训练。"
    echo "       例: bash process_data.sh ${BENCH_NAME} ${CKPT_NAME} ${ENV_CFG_TYPE} ${ACTION_TYPE}"
fi

# --------------------------------------------------------------------------
# 2. 调用官方 train.sh 触发多任务联合训练
# --------------------------------------------------------------------------
echo "[INFO] 启动多任务 cotrain 训练..."
bash train.sh "${BENCH_NAME}" "${CKPT_NAME}" "${ENV_CFG_TYPE}" "${ACTION_TYPE}" "${SEED}" "${GPU_ID}"

echo "============================================================"
echo "[DONE] 训练完成。"
echo "       产物目录: $(pwd)/checkpoints/${BENCH_NAME}-${CKPT_NAME}-${ENV_CFG_TYPE}-${ACTION_TYPE}-${SEED}/"
echo "       下一步:    将该绝对路径填入 policy/UnifiedACT/deploy.yml::ckpt_dir"
echo "============================================================"

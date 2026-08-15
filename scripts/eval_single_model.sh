#!/bin/bash
# Compliant single-model policy server launcher.
# Regardless of the per-task ckpt_name the eval harness passes, this ALWAYS
# loads the one unified cotrain ACT checkpoint, so inference never switches
# models across tasks (satisfies the no-per-task-model-switching rule).
set -euo pipefail
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

# The single unified model — same for every task.
CKPT_DIR=/root/ckpts/RoboDojo-cotrain-arx_x5-joint-0

SCRIPT_DIR=/mnt/workspace/RoboDojo/XPolicyLab/policy/ACT
XPL_ROOT=/mnt/workspace/RoboDojo/XPolicyLab
UTILS_DIR="${XPL_ROOT}/utils"

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate act 2>/dev/null || true

action_dim=$(
    PYTHONPATH="${XPL_ROOT}" python -c "
import sys
from XPolicyLab.utils.process_data import get_action_dim
print(get_action_dim('${env_cfg_type}'))
")

export ACT_ACTION_DIM="${action_dim}"

echo "[COMPLIANT-SERVER] task=${task_name} -> always loading unified ckpt=${CKPT_DIR}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${CKPT_DIR}" \
            ckpt_dir="${CKPT_DIR}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="ACT" \
            action_type="${action_type}" \
            action_dim="${action_dim}"

#!/bin/bash
set -e
# activate the training env (try common conda locations)
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate act 2>/dev/null || true

cd /mnt/workspace/RoboDojo/XPolicyLab/policy/ACT
export PYTHONPATH=/mnt/workspace/RoboDojo/XPolicyLab
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ADIM=$(python -c "import sys; sys.path.insert(0,'/mnt/workspace/RoboDojo/XPolicyLab'); from XPolicyLab.utils.process_data import get_action_dim; print(get_action_dim('arx_x5'))")
export ACT_ACTION_DIM=$ADIM
echo "ACT_ACTION_DIM=$ADIM"

NUM_EPOCHS=${1:-6000}
BATCH=${2:-16}

python3 imitate_episodes.py \
  --bench_name RoboDojo --task_name cotrain --ckpt_setting RoboDojo-cotrain-arx_x5-joint \
  --ckpt_dir /root/ckpts/RoboDojo-cotrain-arx_x5-joint-0 \
  --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 \
  --batch_size $BATCH --dim_feedforward 3200 --num_epochs $NUM_EPOCHS \
  --lr 1e-5 --save_freq 500 --seed 0 --temporal_agg

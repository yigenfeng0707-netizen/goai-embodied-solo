#!/bin/bash
# ==========================================================================
# build_parallel.sh — 并行构建 cotrain 多任务合并数据集
# --------------------------------------------------------------------------
# 修复说明（2026-08-19）：
#   旧版 OFF 表只声明 6 个任务，导致 6 个评测任务完全无训练数据
#   （stack_bowls / fold_clothes / make_toast / arrange_largest_number /
#    store_laptop_and_headphones / stack_blocks），评测全部 0%。
#   现补齐到 12 个评测任务，max_per_task 从 60 提高到 100。
#
#   注：store_laptop_and_headphones 在原始数据集中可能不存在
#   （见 project_memory：该任务无数据集，旧方案用 hang_mugs fallback）。
#   build_cotrain.py 会打印 SKIP_NO_RAW 跳过，不影响其他任务。
# ==========================================================================
export PYTHONPATH=/mnt/workspace/RoboDojo/XPolicyLab
cd /mnt/workspace/RoboDojo/XPolicyLab/policy/ACT

# 12 个评测任务，每任务 offset 间隔 100（max_per_task=100）
# episode_id 全局唯一即可，存在空隙不影响训练。
declare -A OFF=(
  [hang_mugs]=0
  [pack_objects_into_box]=100
  [pour_liquid_into_cup]=200
  [push_T]=300
  [sort_nesting_dolls_by_size]=400
  [sweep_blocks]=500
  [stack_bowls]=600
  [fold_clothes]=700
  [make_toast]=800
  [arrange_largest_number]=900
  [store_laptop_and_headphones]=1000
  [stack_blocks]=1100
)
for t in "${!OFF[@]}"; do
  nohup python /root/build_cotrain.py --task "$t" --offset "${OFF[$t]}" --max_per_task 100 > /root/build_${t}.log 2>&1 &
done
echo "launched ${#OFF[@]} workers at $(date)"
wait
echo "ALL_BUILD_DONE at $(date)"
python - <<'PY'
import json, glob
c = {}
tot = 0
for f in glob.glob('/root/RoboDojo_cotrain/arx_x5-joint/_task_*.json'):
    d = json.load(open(f))
    c[d['task']] = d['count']
    tot += d['count']
json.dump({'counts': c, 'total': tot}, open('/root/RoboDojo_cotrain/arx_x5-joint/_manifest.json', 'w'), indent=2)
print('MANIFEST total', tot, c)
# 校验：至少应覆盖 11 个任务（store_laptop_and_headphones 可能 SKIP）
missing = [t for t in [
    'hang_mugs','pack_objects_into_box','pour_liquid_into_cup','push_T',
    'sort_nesting_dolls_by_size','sweep_blocks','stack_bowls','fold_clothes',
    'make_toast','arrange_largest_number','stack_blocks'
] if t not in c or c[t] == 0]
if missing:
    print('WARNING: missing tasks (no data built):', missing)
PY

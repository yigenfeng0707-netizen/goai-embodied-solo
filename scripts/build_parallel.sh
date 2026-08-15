#!/bin/bash
export PYTHONPATH=/mnt/workspace/RoboDojo/XPolicyLab
cd /mnt/workspace/RoboDojo/XPolicyLab/policy/ACT
declare -A OFF=( [hang_mugs]=0 [pack_objects_into_box]=60 [pour_liquid_into_cup]=120 [push_T]=180 [sort_nesting_dolls_by_size]=240 [sweep_blocks]=300 )
for t in "${!OFF[@]}"; do
  nohup python /root/build_cotrain.py --task "$t" --offset "${OFF[$t]}" --max_per_task 60 > /root/build_${t}.log 2>&1 &
done
echo "launched 6 workers at $(date)"
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
PY

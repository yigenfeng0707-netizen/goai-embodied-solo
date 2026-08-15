#!/bin/bash
# Autonomous pipeline on /root (writable). After cotrain build to /root:
#  1) free NFS quota by deleting some raw episodes (data already in /root cotrain)
#  2) patch NFS TASK_CONFIGS.json with the cotrain entry (absolute /root dataset_dir)
#  3) train the single unified cotrain model (ckpt on /root)
set -e
CT=/root/RoboDojo_cotrain/arx_x5-joint
MAN=$CT/_manifest.json
echo "[pipe $(date)] waiting for build manifest..."
while [ ! -f "$MAN" ]; do sleep 30; done
echo "[pipe] build done: $(cat $MAN)"

# Free NFS project quota: delete a handful of raw episodes (already copied into cotrain).
RAW=/mnt/workspace/RoboDojo/data/GOAI-2026-hf/data/hdf5
for t in hang_mugs pack_objects_into_box pour_liquid_into_cup push_T sort_nesting_dolls_by_size sweep_blocks; do
  d=$RAW/$t/arx_x5/data
  if [ -d "$d" ]; then
    find "$d" -name 'episode_*.hdf5' | sort | tail -6 | xargs -r rm -f
  fi
done
echo "[pipe] freed some raw episodes to relieve NFS quota"

python3 /root/patch_config.py

echo "[pipe] starting training (logging to /root/train_cotrain.log)"
bash /root/train_cotrain.sh "${1:-6000}" "${2:-16}" >> /root/train_cotrain.log 2>&1
echo "[pipe] training finished rc=$?"

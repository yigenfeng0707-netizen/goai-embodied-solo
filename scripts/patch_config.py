import json, os
p = '/mnt/workspace/RoboDojo/XPolicyLab/policy/ACT/TASK_CONFIGS.json'
cfg = json.load(open(p))
man = json.load(open('/root/RoboDojo_cotrain/arx_x5-joint/_manifest.json'))
key = 'RoboDojo-cotrain-arx_x5-joint'
cfg[key] = {
    'dataset_dir': '/root/RoboDojo_cotrain/arx_x5-joint',
    'num_episodes': man['total'],
    'episode_len': 5000,
    'camera_names': ['cam_head', 'cam_right_wrist', 'cam_left_wrist'],
    'policy_context': 'unified',
    'task_index': None,
}
json.dump(cfg, open(p, 'w'), indent=2)
print('patched', key, 'num_episodes', man['total'], 'counts', man['counts'])

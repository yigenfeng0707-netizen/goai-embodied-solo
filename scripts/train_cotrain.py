import os, sys
sys.path.insert(0, '/root/ACT_train')
os.environ['ACT_ACTION_DIM'] = '14'
os.environ['MUJOCO_GL'] = 'egl'
from imitate_episodes import main

task_name = 'RoboDojo-cotrain-arx_x5-joint'


def main_entry():
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    args = {
        'ckpt_dir': '/root/ckpts/' + task_name + '-0',
        'ckpt_setting': task_name,
        'policy_class': 'ACT',
        'onscreen_render': False,
        'batch_size': batch_size,
        'lr': 1e-4,
        'chunk_size': 50,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'num_epochs': num_epochs,
        'seed': 0,
        'temporal_agg': False,
        'save_freq': 500,
    }
    main(args)


if __name__ == '__main__':
    main_entry()

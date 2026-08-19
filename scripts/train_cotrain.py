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
        # lr 统一为 1e-5（与 train_cotrain.sh / configs/train_act.yaml 一致）。
        # 旧值 1e-4 对 ACT VAE+Transformer 偏高，多任务小数据易训练不稳。
        'lr': 1e-5,
        'chunk_size': 50,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'num_epochs': num_epochs,
        'seed': 0,
        # temporal_agg 必须与 deploy.yml 一致（推理期不可变更）。
        'temporal_agg': True,
        'save_freq': 500,
    }
    main(args)


if __name__ == '__main__':
    main_entry()

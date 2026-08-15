import os, sys, json, argparse
sys.path.insert(0, '/mnt/workspace/RoboDojo/XPolicyLab')
import numpy as np
import h5py
import cv2
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import pack_robot_state, decode_image_bit, get_robot_action_dim_info

RAW_ROOT = '/mnt/workspace/RoboDojo/data/GOAI-2026-hf/data/hdf5'
SAVE_DIR = '/root/RoboDojo_cotrain/arx_x5-joint'
ENV = 'arx_x5'
ACTION_TYPE = 'joint'
CAMERAS = ['cam_head', 'cam_right_wrist', 'cam_left_wrist']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--offset', type=int, required=True, help='global start index')
    ap.add_argument('--max_per_task', type=int, default=0)
    args = ap.parse_args()

    info = get_robot_action_dim_info(ENV)
    os.makedirs(SAVE_DIR, exist_ok=True)
    rd = os.path.join(RAW_ROOT, args.task, ENV, 'data')
    if not os.path.isdir(rd):
        print('SKIP_NO_RAW', args.task, flush=True)
        return
    eps = sorted([f for f in os.listdir(rd)
                  if f.startswith('episode_') and f.endswith('.hdf5')])
    if args.max_per_task > 0:
        eps = eps[:args.max_per_task]
    c = 0
    for i, ep in enumerate(eps):
        try:
            data = load_hdf5(os.path.join(rd, ep))
        except Exception as e:
            print('BAD', args.task, ep, e, flush=True)
            continue
        sa = pack_robot_state(data, ACTION_TYPE, info, source_type='dataset', state_type='state')
        aa = pack_robot_state(data, ACTION_TYPE, info, source_type='dataset', state_type='action')
        T = sa.shape[0]
        qpos = []
        acts = []
        cams = {cm: [] for cm in CAMERAS}
        for j in range(T):
            qpos.append(sa[j].astype(np.float32))
            for cm in CAMERAS:
                bit = data['vision'][cm]['colors'][j]
                im = decode_image_bit(bit)
                im = cv2.resize(im, (640, 480))
                cams[cm].append(im)
            acts.append(aa[j])
        gid = args.offset + i
        out = os.path.join(SAVE_DIR, f'episode_{gid:07d}.hdf5')
        with h5py.File(out, 'w') as f:
            f.create_dataset('action', data=np.array(acts, dtype=np.float32))
            ob = f.create_group('observations')
            ob.create_dataset('qpos', data=np.array(qpos, dtype=np.float32))
            im = ob.create_group('images')
            for cm in CAMERAS:
                im.create_dataset(cm, data=np.stack(cams[cm]).astype(np.uint8),
                                  compression='gzip', compression_opts=1)
        c += 1
    with open(os.path.join(SAVE_DIR, f'_task_{args.task}.json'), 'w') as f:
        json.dump({'task': args.task, 'offset': args.offset, 'count': c}, f)
    print('DONE', args.task, 'count', c, 'offset', args.offset, flush=True)


if __name__ == '__main__':
    main()

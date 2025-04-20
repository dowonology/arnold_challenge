import shutil
import hydra
import logging
import numpy as np
import copy
import os
from pathlib import Path
from environment.runner_utils import get_simulation
simulation_app, simulation_context, _ = get_simulation(headless=True, gpu_id=0)

from tasks_make_data import load_task
from utils.env import get_obs_make_data, get_obs

logger = logging.getLogger(__name__)

def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir() and item.suffix == ".npz"])
    data = []
    fnames = []
    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
        fnames.append(npz_path)
    return data, fnames

@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    scene_resolution = (512,512)
    cfg.checkpoint_dir = cfg.checkpoint_dir.split(os.path.sep)
    cfg.checkpoint_dir[-2] = cfg.checkpoint_dir[-2].replace('eval', 'train')
    cfg.checkpoint_dir = os.path.sep.join(cfg.checkpoint_dir)

    render = cfg.visualize
    use_gt = [1, 1]  # Default to [1, 1] as specified
    eval_setting = '2gt'  # Since use_gt=[1,1]

    # 수정된 데이터를 저장할 경로를 설정합니다.
    only_success_data_path = os.path.join(cfg.data_root, cfg.make_name)
    os.makedirs(only_success_data_path, exist_ok=True)

    if cfg.task != 'multi':
        task_list = [cfg.task]
    else:
        task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]

    for task in task_list:
        file_path = os.path.join(cfg.output_folder, f'{task}.txt')
        for eval_split in cfg.eval_splits:
            if os.path.exists(os.path.join(cfg.data_root, task, eval_split)):
                logger.info(f'Evaluating {task} {eval_split}')
                data, fnames = load_data(data_path=os.path.join(cfg.data_root, task, eval_split))
            else:
                logger.info(f'path = {os.path.join(cfg.data_root, task, eval_split)} not exist')
                logger.info(f'{eval_split} not exist')
                continue
            correct = 0
            total = 0
            stats = {}
            while len(data) > 0:
                make_npz_data = []
                anno = data.pop(0)
                fname = fnames.pop(0)
                gt_frames = anno['gt']
                robot_base = gt_frames[0]['robot_base']

                gt_actions = [gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world']]
                if gt_frames[3]['position_rotation_world'] is not None:
                    gt_actions.append(
                        gt_frames[3]['position_rotation_world'] if 'water' not in task
                        else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
                    )
                else:
                    gt_actions.append(None)

                assert gt_actions[0] is not None and gt_actions[1] is not None, "Use first gt action but it is missing"
                assert gt_actions[2] is not None, "Use second gt action but it is missing"

                env, object_parameters, robot_parameters, scene_parameters = load_task(cfg.asset_root, npz=anno, cfg=cfg)

                obs = env.reset(robot_parameters, scene_parameters, object_parameters,
                                robot_base=robot_base, gt_actions=gt_actions, scene_resolution=scene_resolution)

                if cfg.make_data:
                    diff_data = {'diff': env.checker.get_diff()}
                    make_npz_data.append(copy.deepcopy({**obs, **get_obs_make_data(env.robot, env.c_controller), **diff_data}))

                if cfg.record:
                    env.recorder.start_record(
                        traj_dir=os.path.join(cfg.exp_dir, f'traj_{eval_setting}', os.path.split(fname)[-1]),
                        checker=env.checker,
                    )

                try:
                    for i in range(2):
                        if cfg.make_data:
                            obs, suc = env.step(act_pos=None, act_rot=None, render=render, use_gt=True, file_path=file_path, franka=env.robot, cspace_controller=env.c_controller, make_npz_data=make_npz_data, target_frame_num=cfg.target_frame_num)
                        else:
                            obs, suc = env.step(act_pos=None, act_rot=None, render=render, use_gt=True)

                        if suc == -1:
                            break

                except:
                    suc = -1

                env.stop()

                if suc == 1:
                    correct += 1
                else:
                    logger.info(f'{fname}: {suc}')

                total += 1
                log_str = f'correct: {correct} | total: {total} | remaining: {len(data)}'
                logger.info(f'{log_str}\n')
                
                stats[fname] = suc

                if cfg.make_data and suc == 1 and len(make_npz_data) > 0:
                    # Create a new npz file based solely on make_npz_data
                    npz_data_to_save = {'gt': make_npz_data}
                    task_success_path = os.path.join(only_success_data_path, task, eval_split)
                    os.makedirs(task_success_path, exist_ok=True)
                    success_fname = os.path.join(task_success_path, os.path.basename(fname))
                    np.savez_compressed(success_fname, **npz_data_to_save)
                    print(f"New npz file created: {success_fname}")

            logger.info(f'{task} {eval_split} score: {correct/total*100:.2f}\n\n')

    simulation_app.close()


if __name__ == '__main__':
    main()

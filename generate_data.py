# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import pickle
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.tracking.utils import category_to_tracking_name

from utils.data_util import NuScenesClasses

def simpletrack_nms(frame_det_data, iou_threshold=0.1):
    from SimpleTrack.data_loader.nuscenes_loader import nu_array2mot_bbox
    from mot_3d.preprocessing import nms

    boxes = np.concatenate([frame_det_data['translation'],
                            frame_det_data['size'],
                            frame_det_data['rotation'],
                            np.expand_dims(frame_det_data['score'], axis=1)], 
                            axis=1)
    classes = frame_det_data['class']
    boxes_mot = [nu_array2mot_bbox(b) for b in boxes]

    index, _ = nms(boxes_mot, classes, iou_threshold)

    frame_det_data['translation'] = frame_det_data['translation'][index]
    frame_det_data['size'] = frame_det_data['size'][index]
    frame_det_data['yaw'] = frame_det_data['yaw'][index]
    frame_det_data['rotation'] = frame_det_data['rotation'][index]
    frame_det_data['velocity'] = frame_det_data['velocity'][index]
    frame_det_data['class'] = frame_det_data['class'][index]
    frame_det_data['score'] = frame_det_data['score'][index]

    return frame_det_data

def write_data(data, output):
    print('Writing data into pkl files...')
    for scene in tqdm(data):
        for i, frame in enumerate(scene):
            if i < 1:
                scene_id = frame['scene_id']
                scene_output = output / f'{scene_id:04d}'
                scene_output.mkdir(parents=True, exist_ok=True)
            
            frame_id = frame['frame_id']
            filename = scene_output / f'{frame_id:03d}.pkl'

            content = {'dets': frame['detections'],
                       'gts': frame['ground_truths'],
                       'num_dets': frame['num_dets'],
                       'num_gts': frame['num_gts'],
                       'ego_translation': frame['ego_translation'],
                       'timestamp': frame['timestamp'],
                       'token': frame['sample_token']
                      }

            with open(filename, 'wb') as f:
                pickle.dump(content, f)

def generate_nusc_seq_data(nusc, det_boxes, scenes, sequences_by_name, apply_nms=False):

    print('Generating detection and ground truth sequences...')
    result = []

    for scene_id, scene_name in enumerate(tqdm(scenes)):
        scene = sequences_by_name[scene_name]
        first_token = scene['first_sample_token']
        last_token = scene['last_sample_token']
        current_token = first_token

        scene_result = []
        tracking_id_set = set()

        frame_id = 0
        while True:
            current_sample = nusc.get('sample', current_token)

            # Get ego pose data
            lidar_top_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
            ego_trans = np.array(ego_pose['translation'], dtype=np.float32)
            ego_timestamp = np.array(ego_pose['timestamp'], dtype=np.int)

            ## Process and concat detections for every frame
            frame_dets = det_boxes[current_token]
            det_trans = []
            det_size = []
            det_yaw = []
            det_rot = []
            det_velo = []
            det_class = []
            det_score = []

            for det in frame_dets:
                det_dict = det.serialize()
                if det_dict['detection_name'] in NuScenesClasses.keys():
                    det_trans.append(det_dict['translation'])
                    det_size.append(det_dict['size'])
                    det_yaw.append([quaternion_yaw(Quaternion(det_dict['rotation']))])
                    det_rot.append(det_dict['rotation'])
                    det_velo.append(det_dict['velocity'])
                    det_class.append(NuScenesClasses[det_dict['detection_name']])
                    det_score.append(det_dict['detection_score'])

            frame_dets_dict = {
                'translation': np.array(det_trans, dtype=np.float32), # [N, 3]
                'size': np.array(det_size, dtype=np.float32), # [N, 3]
                'yaw': np.array(det_yaw, dtype=np.float32), # [N, 1]
                'rotation': np.array(det_rot, dtype=np.float32), # [N, 4]
                'velocity': np.array(det_velo, dtype=np.float32), # [N, 2]
                'class': np.array(det_class, dtype=np.int32), # [N]
                'score': np.array(det_score, dtype=np.float32), # [N]
            }

            if apply_nms:
                frame_dets_dict = simpletrack_nms(frame_dets_dict, iou_threshold=0.1)
            
            ## Process and concat ground truths for every frame
            frame_ann_tokens = current_sample['anns']
            gt_trans = []
            gt_size = []
            gt_yaw = []
            gt_rot = []
            gt_class = []
            gt_track_token = []

            gt_next_exist = []
            gt_next_trans = []
            gt_next_size = []
            gt_next_yaw = []

            for ann_token in frame_ann_tokens:
                ann = nusc.get('sample_annotation', ann_token)
                tracking_name = category_to_tracking_name(ann['category_name'])
                if tracking_name is not None:
                    instance_token = ann['instance_token']
                    tracking_id_set.add(instance_token)

                    gt_trans.append(ann['translation'])
                    gt_size.append(ann['size'])
                    gt_yaw.append([quaternion_yaw(Quaternion(ann['rotation']))])
                    gt_rot.append(ann['rotation'])
                    gt_class.append(NuScenesClasses[tracking_name])
                    gt_track_token.append(instance_token)

                    next_ann_token = ann['next']
                    if next_ann_token == "":
                        gt_next_exist.append(False)
                        gt_next_trans.append([0.0, 0.0, 0.0])
                        gt_next_size.append([0.0, 0.0, 0.0])
                        gt_next_yaw.append([0.0])
                    else:
                        gt_next_exist.append(True)
                        next_ann = nusc.get('sample_annotation', next_ann_token)
                        gt_next_trans.append(next_ann['translation'])
                        gt_next_size.append(next_ann['size'])
                        gt_next_yaw.append([quaternion_yaw(Quaternion(next_ann['rotation']))])

            frame_anns_dict = {
                'translation': np.array(gt_trans, dtype=np.float32), # [M, 3]
                'size': np.array(gt_size, dtype=np.float32), # [M, 3]
                'yaw': np.array(gt_yaw, dtype=np.float32), # [M, 1]
                'rotation': np.array(gt_rot, dtype=np.float32), # [M, 4]
                'class': np.array(gt_class, dtype=np.int32), # [M]
                'tracking_id': gt_track_token, # [M]
                'next_exist': np.array(gt_next_exist, dtype=np.bool), # [M]
                'next_translation': np.array(gt_next_trans, dtype=np.float32), # [M, 3]
                'next_size': np.array(gt_next_size, dtype=np.float32), # [M, 3]
                'next_yaw': np.array(gt_next_yaw, dtype=np.float32), # [M, 1]
            }

            frame_result = {'detections': frame_dets_dict,
                            'ground_truths': frame_anns_dict,
                            'num_dets': len(det_trans), # int: N
                            'num_gts': len(gt_trans), # int: M
                            'scene_id': scene_id,
                            'frame_id': frame_id,
                            'ego_translation': ego_trans,
                            'timestamp': ego_timestamp,
                            'sample_token': current_token
                            }
            scene_result.append(frame_result)

            if current_token == last_token:
                break

            next_token = current_sample['next']
            current_token = next_token
            frame_id += 1

        assert len(scene_result) == scene['nbr_samples']
        
        ## Convert instance token to tacking id for the whole scene
        tracking_token_to_id = {}
        for i, tracking_id in enumerate(tracking_id_set):
            tracking_token_to_id.update({tracking_id: i})
        
        for frame_result in scene_result:
            for i, tracking_token in enumerate(frame_result['ground_truths']['tracking_id']):
                tracking_id = tracking_token_to_id[tracking_token]
                frame_result['ground_truths']['tracking_id'][i] = tracking_id
            frame_result['ground_truths']['tracking_id'] = \
                np.array(frame_result['ground_truths']['tracking_id'], dtype=np.int32)

        result.append(scene_result)
    
    print('Done generating.')
    print('======')
    
    return result

def generate_nusc_data(version, dataset_dir, detection_dir, output_dir, apply_nms=False):

    dataset_dir = dataset_dir / version
    train_result_file = detection_dir / "train.json"
    val_result_file = detection_dir / "val.json"
    test_result_file = detection_dir / "test.json"

    version_fullname = version
    if version == "v1.0":
        version_fullname += '-trainval'
    nusc = NuScenes(version=version_fullname, dataroot=dataset_dir, verbose=True)
    nusc_test = NuScenes(version='v1.0-test', dataroot=dataset_dir, verbose=True)
    sequences_by_name = {scene["name"]: scene for scene in nusc.scene}
    sequences_by_name.update({scene["name"]: scene for scene in nusc_test.scene})
    splits_to_scene_names = create_splits_scenes()

    train_split = 'train' if version == "v1.0" else 'mini_train'
    val_split = 'val' if version == "v1.0" else 'mini_val'
    test_split = 'test'
    train_scenes = splits_to_scene_names[train_split]
    val_scenes = splits_to_scene_names[val_split]
    test_scenes = splits_to_scene_names[test_split]

    result_files = [train_result_file, val_result_file]
    scenes = [train_scenes, val_scenes]
    output_dirs = [output_dir / 'training', output_dir / 'validation']

    # Train and validation split
    for result_file, scene, output in zip(result_files, scenes, output_dirs):
        # output.mkdir(parents=True, exist_ok=True)

        print('Loading Nusences 3d detctions...')
        det_boxes, _ = load_prediction(result_file, 10000, DetectionBox, verbose=True)
        print('======')

        # train_data[scene_id][frame_id] = 
        # {'detections': {'box', 'class', 'score'}, 'ground_truths': {'box', 'class', 'tracking_id'}}
        data = generate_nusc_seq_data(nusc, det_boxes, scene, sequences_by_name, apply_nms)

        write_data(data, output)
    
    # Test split
    print('Loading Nusences 3d detctions...')
    det_boxes, _ = load_prediction(test_result_file, 10000, DetectionBox, verbose=True)
    print('======')

    data = generate_nusc_seq_data(nusc_test, det_boxes, test_scenes, sequences_by_name, apply_nms)

    write_data(data, output_dir / 'testing')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Nuscenes data preprocessing')
    args.add_argument('--dataset_dir', default=None, type=str,
                      help='Directory where nuScenes dataset is stored')
    args.add_argument('--version', default="v1.0", type=str,
                      help='Version of nuScenes dataset')
    args.add_argument('--detection_dir', default=None, type=str,
                      help='Directory where detection results are stored')
    args.add_argument('--output_dir', default=None, type=str,
                      help='Directory where preprocessed pickle files will be stored')
    args.add_argument('--apply_nms', action='store_true',
                      help='Whether to apply a Non-Maximum Suppression')
    args = args.parse_args()

    generate_nusc_data(version=args.version,
                       dataset_dir=Path(args.dataset_dir),
                       detection_dir=Path(args.detection_dir),
                       output_dir=Path(args.output_dir),
                       apply_nms=args.apply_nms)
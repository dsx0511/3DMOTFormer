# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from nuScenes devkit (https://github.com/nutonomy/nuscenes-devkit)
# Copyright 2021 Motional. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
from matplotlib.axes import Axes

from nuscenes import NuScenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points


class BoxWrapper(Box):
    def render_box_and_velo(self,
                            axis: Axes,
                            view: np.ndarray = np.eye(3),
                            normalize: bool = False,
                            colors: Tuple = ('b', 'r', 'k'),
                            linewidth: float = 2) -> None:
        
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)

        # Draw predicted position
        velo = self.velocity

        if np.sqrt(np.square(velo[0]) + np.square(velo[1])) >= 0.2:
            axis.arrow(center_bottom[0], center_bottom[1], velo[0] * 0.5, velo[1] * 0.5,
                    color=colors[0], width=0.15, head_width = 0.6)
        

def track_boxes_to_sensor(boxes: List[TrackingBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = BoxWrapper(box.translation, box.size, Quaternion(box.rotation), 
                         velocity=(box.velocity[0], box.velocity[1], 0),
                         name=box.tracking_name, token=box.tracking_id, score=box.tracking_score)

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out

class TrackingEvalWrapper(TrackingEval):
    """
    A wrapper of the nuscenes TrackingEval class with desired functionality.
    """
    def __init__(self,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True,
                 render_classes: List[str] = None):
        super().__init__(config, result_path, eval_set, output_dir, nusc_version, nusc_dataroot, verbose, render_classes)
        self.nusc_version = nusc_version
        self.nusc_dataroot = nusc_dataroot

    def visualize(self,num_vis, eval_range=50, nsweeps=1, score_threshold=0.0):
        print('=====')
        print('Start visualization')
        nusc = NuScenes(version=self.nusc_version, verbose=self.verbose, dataroot=self.nusc_dataroot)

        savepath = self.output_dir / 'vis'
        # self.track_gt: Dict[str, Dict[int, List[TrackingBox]]]:
        for scene_id, scene_token in enumerate(self.tracks_gt.keys()):
            if scene_id >= num_vis:
                break
            
            print(f'Visualizing {scene_id}-th sequence.')
            scene_tracks_gt = self.tracks_gt[scene_token]
            scene_tracks_pred = self.tracks_pred[scene_token]

            savepath_sub = Path(savepath) / scene_token
            savepath_sub.mkdir(parents=True, exist_ok=True)

            scene = nusc.get('scene', scene_token)
            cur_sample_token = scene['first_sample_token']
            last_token = scene['last_sample_token']

            frame_id = 0

            while True:
                # Retrieve sensor & pose records.
                cur_sample = nusc.get('sample', cur_sample_token)
                sd_record = nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])
                timestamp = cur_sample['timestamp']
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

                # Get boxes.
                frame_gt_global = scene_tracks_gt[timestamp]
                frame_pred_global = scene_tracks_pred[timestamp]

                # Map GT boxes to lidar.
                frame_gt = track_boxes_to_sensor(frame_gt_global, pose_record, cs_record)

                # Map EST boxes to lidar.
                frame_pred = track_boxes_to_sensor(frame_pred_global, pose_record, cs_record)

                # Get point cloud in lidar frame.
                pc, _ = LidarPointCloud.from_file_multisweep(nusc, cur_sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

                _, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(30, 15))
                ax_pred.set_title('Prediction')
                ax_gt.set_title('Ground Truth')

                ax_pred.title.set_size(16)
                ax_gt.title.set_size(16)

                # Show point cloud.
                points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
                dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
                colors = np.minimum(1, dists / eval_range)
                ax_pred.scatter(points[0, :], points[1, :], c=colors, s=0.2)
                ax_gt.scatter(points[0, :], points[1, :], c=colors, s=0.2)

                # Show ego vehicle.
                ax_pred.plot(0, 0, 'x', color='black')
                ax_gt.plot(0, 0, 'x', color='black')

                # Show GT boxes.
                for box in frame_gt:
                    color = (float(hash(box.token + 'r') % 256) / 255,
                             float(hash(box.token + 'g') % 256) / 255,
                             float(hash(box.token + 'b') % 256) / 255)
                    box.render_box_and_velo(ax_gt, view=np.eye(4), colors=(color, color, color), linewidth=3)

                # Show EST boxes.
                for box in frame_pred:
                    color = (float(hash(box.token + 'r') % 256) / 255,
                             float(hash(box.token + 'g') % 256) / 255,
                             float(hash(box.token + 'b') % 256) / 255)
                    if box.score > score_threshold:
                        box.render_box_and_velo(ax_pred, view=np.eye(4), colors=(color, color, color), linewidth=3)

                # Limit visible range.
                axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
                ax_pred.set_xlim(-axes_limit, axes_limit)
                ax_pred.set_ylim(-axes_limit, axes_limit)
                ax_gt.set_xlim(-axes_limit, axes_limit)
                ax_gt.set_ylim(-axes_limit, axes_limit)

                plt.savefig(savepath_sub / f'{frame_id:03d}.png', bbox_inches='tight')
                plt.close()

                if cur_sample_token == last_token:
                    break
                
                cur_sample_token = cur_sample['next']
                frame_id += 1


def eval_nusc_tracking(res_path, eval_set="val", output_dir=None, root_path=None, verbose=True,
                       num_vis=10, vis_only=False, vis_score_thresh=0.0, eval_range=50.0):
    
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEvalWrapper(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
        nusc_version="v1.0-trainval",
        nusc_dataroot=root_path,
        # render_classes=["bus", "car"],
    )
    if num_vis > 0:
        nusc_eval.visualize(num_vis, score_threshold=vis_score_thresh, eval_range=eval_range)
    if not vis_only:
        metrics_summary = nusc_eval.main()
        return metrics_summary
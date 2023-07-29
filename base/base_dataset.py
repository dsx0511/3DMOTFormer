# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Pytorch Template Project (https://github.com/victoresque/pytorch-template)
# Copyright (c) 2018 Victor Huang. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import glob
import pickle
from tqdm import tqdm
from abc import abstractmethod

from scipy.optimize import linear_sum_assignment

from utils.data_util import NuScenesClasses


class BaseDataset(Dataset):
    def __init__(self, data_dir, 
                       split, 
                       dataset, 
                       iou_matching=False, 
                       score_threshold=0.0):

        self.split = split
        self.data_dir = Path(data_dir) / self.split

        if dataset == "nuscenes":
            self.num_classes = len(NuScenesClasses)
        else:
            raise ValueError("Unsupported dataset: ", dataset)
        
        self.iou_matching = iou_matching
        self.score_threshold = score_threshold

        self.data = []
        self.nbr_samples = []

        scenes = sorted(os.listdir(self.data_dir))
        for scene in scenes:
            scene_dir = self.data_dir / scene
            frame_files = sorted(glob.glob(str(scene_dir) + '/*.pkl'))
            self.nbr_samples.append(len(frame_files))
            scene_data = []
            scene_id = int(scene)

            for frame_file in frame_files:
                with open(frame_file, 'rb') as f:
                    frame_data = pickle.load(f)
                    frame_data = self._to_torch_tensor(frame_data)
                    frame_data = self._build_bounding_box(frame_data)
                    if self.score_threshold > 0.0:
                        frame_data = self._score_filter(frame_data, threshold=self.score_threshold)
                    frame_data['scene_id'] = scene_id
                                            
                scene_data.append(frame_data)

            self.data.append(scene_data)

        self._assign_track_id()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    def _score_filter(self, data, threshold=0.1):
        '''
        Filter detection boxes according to their scores.
        '''
        det_data = data['dets']
        score_mask = det_data['score'] >= threshold
        det_data_filtered = {}
        for field_name, field in det_data.items():
            filtered_field = field[score_mask]
            det_data_filtered[field_name] = filtered_field
        data['dets'] = det_data_filtered
        return data

    def _to_torch_tensor(self, data):
        '''
        Converting numpy array to torch tensor.
        '''
        data['dets']['translation'] = torch.from_numpy(data['dets']['translation'])
        data['dets']['size'] = torch.from_numpy(data['dets']['size'])
        data['dets']['yaw'] = torch.from_numpy(data['dets']['yaw'])
        data['dets']['velocity'] = torch.from_numpy(data['dets']['velocity'])
        data['dets']['class'] = torch.from_numpy(data['dets']['class'])
        data['dets']['score'] = torch.from_numpy(data['dets']['score'])

        data['gts']['translation'] = torch.from_numpy(data['gts']['translation'])
        data['gts']['size'] = torch.from_numpy(data['gts']['size'])
        data['gts']['yaw'] = torch.from_numpy(data['gts']['yaw'])
        data['gts']['class'] = torch.from_numpy(data['gts']['class'])
        data['gts']['tracking_id'] = torch.from_numpy(data['gts']['tracking_id'])
        data['gts']['next_exist'] = torch.from_numpy(data['gts']['next_exist'])
        data['gts']['next_translation'] = torch.from_numpy(data['gts']['next_translation'])
        data['gts']['next_size'] = torch.from_numpy(data['gts']['next_size'])
        data['gts']['next_yaw'] = torch.from_numpy(data['gts']['next_yaw'])

        return data
    
    def _build_bounding_box(self, data):
        '''
        Converting bounding box to [x, y, z, w, l, h, yaw] format
        '''
        data['dets']['box'] = torch.cat([data['dets']['translation'],
                                         data['dets']['size'],
                                         data['dets']['yaw']], dim=1).view(-1, 7)

        data['gts']['box'] = torch.cat([data['gts']['translation'],
                                        data['gts']['size'],
                                        data['gts']['yaw']], dim=1).view(-1, 7)
        return data
    
    def _dets_gt_matching(self, det_box, det_cls, gt_box, gt_cls):
        '''
        Assign tracking ID for detection boxes in data preprocessing.
        '''
        cls_valid_mask = torch.eq(det_cls.unsqueeze(1), gt_cls.unsqueeze(0))

        if self.iou_matching:
            # from ops.iou3d import iou3d_nms_utils
            # iou = iou3d_nms_utils.boxes_iou_bev_cpu(det_box, gt_box)
            # iou_gpu = iou3d_nms_utils.boxes_iou_bev(det_box.cuda(), gt_box.cuda())

            import iou3d_nms_cuda
            assert det_box.shape[1] == gt_box.shape[1] == 7
            iou = torch.FloatTensor(torch.Size((det_box.shape[0], gt_box.shape[0]))).zero_()

            iou3d_nms_cuda.boxes_iou_bev_cpu(det_box.contiguous(), gt_box.contiguous(), iou)


            iou_valid_mask = iou > 0
            valid_mask = torch.logical_and(cls_valid_mask, iou_valid_mask)
            invalid_mask = torch.logical_not(valid_mask)
            cost = - iou + 1e18 * invalid_mask
        else:
            center_dist = torch.cdist(det_box[:, :2], gt_box[:, :2], p=2.0)
            dist_valid_mask = torch.le(center_dist, 2.0)
            valid_mask = torch.logical_and(cls_valid_mask, dist_valid_mask)
            invalid_mask = torch.logical_not(valid_mask)
            cost = center_dist + 1e18 * invalid_mask

        cost[cost > 1e16] = 1e18

        # row_ind: index of detection boxes
        # col_ind: index of ground truth
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1e16:
                matches.append([i, j])
        
        return matches

    def _assign_track_id(self):
        '''
        This function matches detection boxes to annotated boxes to assign the object ID.
        The matching has to be a bipartite matching.
        For a matched pairs, the detection boxes will get the object IDs of the GT box.
        For unmatched detections, their object IDs are set to -1.
        '''

        print('Applying matching between detection and gt boxes...')

        num_all_gts = 0
        num_all_matched_gts = 0

        for sequence in tqdm(self.data):
            for seq_id, frame in enumerate(sequence):
                scene_id = frame['scene_id']
                # Calculate class-specific IOU, only boxes with same class overlap
                det_box = frame['dets']['box']
                gt_box = frame['gts']['box']

                det_cls = frame['dets']['class']
                gt_cls = frame['gts']['class']

                # Avoid tracking id match across batch elements
                gt_track_id = frame['gts']['tracking_id'] + scene_id * 1000
                gt_next_exist = frame['gts']['next_exist']
                get_next_trans = frame['gts']['next_translation']
                get_next_trans = get_next_trans.view(get_next_trans.size(0), 3)
                gt_next_x = get_next_trans[:, 0]
                gt_next_y = get_next_trans[:, 1]

                matches = self._dets_gt_matching(det_box, det_cls, gt_box, gt_cls)

                num_all_gts += frame['num_gts']
                num_all_matched_gts += len(matches)

                matches = torch.Tensor(matches).view(-1, 2).long()
                num_det = det_box.size(0)
                # For unmatched dets, the value of the matching are -1, otherwise are the matched track ID
                det_tracking_id = - torch.ones(num_det, dtype=torch.int)
                # Given the index (of tensor dimension) of matched GTs, find their corresponding track ID
                matched_tracking_id = gt_track_id[matches[:, 1]]
                # Assign matched dets with the corrspoinding track ID of matched GTs' index
                det_tracking_id[matches[:, 0]] = matched_tracking_id

                # The target of the location in the next frame is also assigned based on the matching result
                matched_gt_next_exist = gt_next_exist[matches[:, 1]]
                matched_gt_next_x = gt_next_x[matches[:, 1]]
                matched_gt_next_y = gt_next_y[matches[:, 1]]

                det_next_exist = torch.zeros(num_det, dtype=torch.bool)
                det_next_exist[matches[:, 0]] = matched_gt_next_exist

                det_next_x = torch.zeros(num_det, dtype=torch.float)
                det_next_y = torch.zeros(num_det, dtype=torch.float)
                det_next_x[matches[:, 0]] = matched_gt_next_x
                det_next_y[matches[:, 0]] = matched_gt_next_y

                det_next_trans = torch.stack([det_next_x, det_next_y], dim=1)
                
                frame.update({'det_matched_track_id': det_tracking_id,
                              'det_next_exist': det_next_exist,
                              'det_next_trans': det_next_trans,
                              })
        
        print(f'{num_all_gts} gt objects are found, {num_all_matched_gts} are matched')

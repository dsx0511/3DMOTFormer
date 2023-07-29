# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch

from torch_geometric.data import Data

from base.base_dataset import BaseDataset
from utils.data_util import torch_one_hot
import utils.graph_util as graph_util


class PatchwiseDataset(BaseDataset):
    def __init__(self, data_dir, split, sample_length, dataset, iou_matching, score_threshold, graph_truncation_dist):
        super().__init__(data_dir, split, dataset, iou_matching, score_threshold)

        assert sample_length >= 2, 'sample length must be grater than or equal to 2'
        self.sample_length = sample_length
        self.graph_truncation_dist = graph_truncation_dist
                
        self.meta = self._generate_meta()
        self.follow_batch = ['x']

    def _generate_meta(self):
        '''
        Generate meta as a list, where each element holds the scene_id and frame_id 
        of the first frame for a training sample.
        '''
        print('Generating dataset meta...')
        meta = []
        num_dropped = 0

        for seq_id, nbr in enumerate(self.nbr_samples):
            for frame_id in range(nbr - self.sample_length + 1):
                # Filter out mini-sequences that doesn't have a track
                # TODO: Maybe include these mini-sequences for a better generalization?
                drop = False
                for i in range(self.sample_length):
                    num_track = self.data[seq_id][frame_id + i]['num_gts']
                    if num_track == 0:
                        drop = True

                if drop:
                    num_dropped += 1
                else:
                    meta.append({'scene_id': seq_id,
                                'frame_id': frame_id})
        
        print(f'Generated {len(meta)}, dropped {num_dropped} mini-sequences')
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        """
        Get a training sample with length `self.sample_length`, where each training sample 
        holds a mini-sequence from a scene in nuscenes train split.

        :param idx: Integer, the index of the dataset.
        :return seq_data: A list with torch_geometric.data.Data instances.
        """
        # Ger the first frame using meta
        seq_id = self.meta[idx]['scene_id']
        frame_id = self.meta[idx]['frame_id']

        seq_data = []

        # Store data of the following self.sample_length frames into a training mini-sequence
        for i in range(self.sample_length):
            detections = self.data[seq_id][frame_id + i]['dets']
            det_matched_track_id = self.data[seq_id][frame_id + i]['det_matched_track_id']
            det_next_exist = self.data[seq_id][frame_id + i]['det_next_exist']
            det_next_trans = self.data[seq_id][frame_id + i]['det_next_trans']
            velo_target = (det_next_trans - detections['box'][:, :2]) * 2.0

            det_box = detections['box']
            det_velo = detections['velocity']
            det_category = detections['class']
            det_class_one_hot = torch_one_hot(det_category, self.num_classes)
            det_score = torch.unsqueeze(detections['score'], 1)
            det_feat = torch.cat([det_box, det_velo, det_class_one_hot, det_score], 1)

            # Build the adjacency matrix of the detection graph
            det_adj = graph_util.bev_euclidean_distance_adj(det_box, det_category, self.graph_truncation_dist)
            edge_index_det = graph_util.adj_to_edge_index(det_adj)

            frame_data = Data(x=det_feat,
                              edge_index=edge_index_det,
                              tracking_id=det_matched_track_id,
                              det_box=det_box,
                              det_velo=det_velo,
                              det_class=det_category,
                              det_score=det_score,
                              frame_id=torch.tensor(frame_id, dtype=torch.int),
                              next_exist=det_next_exist,
                              velo_target=velo_target,
                              )
            seq_data.append(frame_data)
        
        return seq_data
# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from utils.data_util import torch_one_hot
import utils.graph_util as graph_util
from base.base_dataset import BaseDataset


class Sequence(Dataset):
    def __init__(self, data, num_classes, graph_truncation_dist):
        super().__init__()
        self.data = data
        self.num_classes = num_classes
        self.graph_truncation_dist = graph_truncation_dist
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        token = self.data[idx]['token']
        detections = self.data[idx]['dets']
        det_matched_track_id = self.data[idx]['det_matched_track_id']
        det_next_exist = self.data[idx]['det_next_exist']
        det_next_trans = self.data[idx]['det_next_trans']
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
                          next_exist=det_next_exist,
                          velo_target=velo_target,
                          )
        
        return frame_data, token


class SequentialDataset(BaseDataset):
    def __init__(self, data_dir, split, dataset, iou_matching, score_threshold, graph_truncation_dist):
        super().__init__(data_dir, split, dataset, iou_matching, score_threshold)

        self.graph_truncation_dist = graph_truncation_dist
        self.meta = self._generate_meta()

    def _generate_meta(self):
        '''
        Generate meta as a list, where each element holds a Sequence instance which stores the
        data of a sequence in nuscenes validation or test splits
        '''

        print(f'Generating {self.split} dataset meta...')
        meta = []

        for d in self.data:
            meta.append(Sequence(d, self.num_classes, self.graph_truncation_dist))
        
        print(f'Generated {len(meta)} sequences')
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return self.meta[idx]
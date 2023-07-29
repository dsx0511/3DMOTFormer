# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

from pyquaternion import Quaternion

import torch

import utils.graph_util as graph_util
from utils.match_util import dets_tracks_matching
from utils.data_util import NuScenesClasses

def update_field(det_field, track_field, match, new_det, inactive_track,
                 update_with_dets=True):
    if update_with_dets:
        field_mat = det_field[match[:, 1]]
    else:
        field_mat = track_field[match[:, 0]]

    new_det_field = det_field[new_det]
    inactive_track_field = track_field[inactive_track]

    new_track_field = torch.cat([field_mat, new_det_field, inactive_track_field], 0)
    return new_track_field

def update_feature(det_feat, track_feat, match, new_det, inactive_track,
                   weight=0.5):
    feat_mat = weight * det_feat[match[:, 1]] + (1 - weight) * track_feat[match[:, 0]]

    new_det_feat = det_feat[new_det]
    inactive_track_feat = track_feat[inactive_track]

    new_track_feat = torch.cat([feat_mat, new_det_feat, inactive_track_feat], 0)
    return new_track_feat

class Tracker(object):
    def __init__(self, max_age=0, 
                       active_track_thresh=0.1, 
                       feature_update_weight=0.5, 
                       hungarian=True,
                       graph_truncation_dist=5.0):

        self.max_age = max_age
        self.active_track_thresh = active_track_thresh
        self.feature_update_weight = feature_update_weight
        self.hungarian = hungarian
        self.graph_truncation_dist = graph_truncation_dist

        self._initialized = False
        self.reset()

    @property
    def is_initialized(self):
        return self._initialized
    
    def reset(self):
        self.id_count = 0
        # track state are input and output for the network
        self.track_state = {'features': None,
                            'boxes': None,
                            'velo': None,
                            'edge_index': None,
                            'classes': None,
                            'gt_tracking_id': None,
                            'age': None,
                            }
        # track info are for exporting the results
        self.track_info = []
        self._initialized = False

    def init_tracks(self, data):
        '''
        Only for the first frame: Initialize all detections as new tracks.
        '''
        track_feat = data.x
        track_boxes = data.det_box
        track_velo = data.det_velo
        edge_index_track = data.edge_index
        track_class = data.det_class
        track_score = data.det_score
        track_gt = data.tracking_id
        track_age = torch.ones_like(track_class, dtype=torch.int)

        self.track_state = {'features': track_feat,
                            'boxes': track_boxes,
                            'velo': track_velo,
                            'edge_index': edge_index_track,
                            'classes': track_class,
                            'gt_tracking_id': track_gt,
                            'age': track_age
                            }

        num_tracks = track_feat.size(0)
        for i in range(num_tracks):
            yaw = track_boxes[i, 6]
            rotation = Quaternion(axis=[0, 0, 1], angle=yaw.cpu().numpy())
            track = {
                "translation": track_boxes[i, :3].cpu().numpy(),
                "size": track_boxes[i, 3:6].cpu().numpy(),
                "rotation": rotation.elements,
                "velocity": track_velo[i].cpu().numpy(),
                "tracking_id": self.id_count,
                "tracking_name": list(NuScenesClasses.keys())[track_class[i]],
                "tracking_score": track_score[i].cpu().numpy(),
                "age": 1,
                "active": 1,
            }
            self.id_count += 1
            self.track_info.append(track)

        self._initialized = True

    def track_step(self, affinity, pred_velo, inter_graph, data, track_latent_feat, det_latent_feat):
        '''
        Update tracks based on the predicted affinity.
        '''
        # embedding: [box_coder(0-8), one_hot_class(9-15), score(16)]
        # det_init_feat = data.x
        det_boxes = data.det_box
        det_velo = data.det_velo
        det_class = data.det_class
        det_score = data.det_score
        det_gt = data.tracking_id

        edge_index_inter = inter_graph.edge_index
        num_tracks = inter_graph.size_s
        num_dets = inter_graph.size_t
        affinity = affinity.squeeze(1)

        adj = torch.zeros([num_tracks, num_dets],
                           dtype=torch.bool, device=edge_index_inter.device)
        affinity_dense = torch.zeros([num_tracks, num_dets],
                                      dtype=affinity.dtype, device=edge_index_inter.device)

        adj[edge_index_inter[0, :], edge_index_inter[1, :]] = 1
        affinity_dense[edge_index_inter[0, :], edge_index_inter[1, :]] = affinity

        match, unmat_det, unmat_trk = dets_tracks_matching(
            affinity_dense, adj, num_dets, num_tracks,
            active_thresh=self.active_track_thresh, hungarian=self.hungarian)
        
        inactive_trk = []
        for i in unmat_trk:
            if self.track_info[i]['age'] < self.max_age:
                inactive_trk.append(i)

        # update track info
        self._update_track_info(match, unmat_det, inactive_trk, det_boxes,
                                det_velo, det_class, det_score)

        # update track state
        self._update_track_state(match, unmat_det, inactive_trk, track_latent_feat,
                                 det_latent_feat, det_boxes, pred_velo, det_class, det_gt)
        
        assert len(self.track_info) == self.track_state['features'].size(0)


    def _update_track_info(self, matches, new_dets, inactive_tracks,
                           det_boxes, det_velo, det_class, det_score):
        '''
        Update external information for writing results.
        Generate a dictionary with all neccessary information for nuscenes json format.
        '''
        new_track_info = []

        for m in matches:
            yaw = det_boxes[m[1], 6]
            rotation = Quaternion(axis=[0, 0, 1], angle=yaw.cpu().numpy())
            class_id = det_class[m[1]]
            track = {
                "translation": det_boxes[m[1], :3].cpu().numpy(),
                "size": det_boxes[m[1], 3:6].cpu().numpy(),
                "rotation": rotation.elements,
                "velocity": det_velo[m[1], :].cpu().numpy(),
                "tracking_id": self.track_info[m[0]]["tracking_id"],
                "tracking_name": list(NuScenesClasses.keys())[class_id],
                "tracking_score": det_score[m[1]].cpu().numpy(),
                "age": 1,
                "active": self.track_info[m[0]]["active"] + 1,
            }
            new_track_info.append(track)
        
        for i in new_dets:
            yaw = det_boxes[i, 6]
            rotation = Quaternion(axis=[0, 0, 1], angle=yaw.cpu().numpy())
            class_id = det_class[i]
            track = {
                "translation": det_boxes[i, :3].cpu().numpy(),
                "size": det_boxes[i, 3:6].cpu().numpy(),
                "rotation": rotation.elements,
                "velocity": det_velo[i, :].cpu().numpy(),
                "tracking_id": self.id_count,
                "tracking_name": list(NuScenesClasses.keys())[class_id],
                "tracking_score": det_score[i].cpu().numpy(),
                "age": 1,
                "active": 1,
            }
            self.id_count += 1
            new_track_info.append(track)
        
        for j in inactive_tracks:
            track = self.track_info[j]
            track["translation"][:2] += track["velocity"] * 0.5
            track["age"] += 1
            track["active"] = 0
            new_track_info.append(track)
                
        self.track_info = new_track_info

    def _update_track_state(self, matches, new_dets, inactive_tracks, track_latent_feat,
                            det_latent_feat, det_boxes, pred_velo, det_class, det_gt):
        '''
        Update internal information for the network.
        This function is very similar to `Trainer._batch_update_tracks` function.
        A track instance is assigned with feature, box, velocity, category and matched object ID,
        and this function update these fields using different rules.
        '''
        track_boxes = self.track_state['boxes']
        track_velo = self.track_state['velo']
        track_class = self.track_state['classes']
        track_gt = self.track_state['gt_tracking_id']
        track_age = self.track_state['age']

        new_track_feat = update_feature(det_latent_feat, track_latent_feat, matches, new_dets,
                                      inactive_tracks, weight=self.feature_update_weight)
        new_track_boxes = update_field(det_boxes, track_boxes, matches, new_dets,
                                      inactive_tracks, update_with_dets=True)
        new_track_velo = update_field(pred_velo, track_velo, matches, new_dets,
                                      inactive_tracks, update_with_dets=True)
        new_track_class = update_field(det_class, track_class, matches, new_dets,
                                      inactive_tracks, update_with_dets=True)
        new_track_gt = update_field(det_gt, track_gt, matches, new_dets,
                                      inactive_tracks, update_with_dets=True)

        det_age = torch.ones(len(matches) + len(new_dets), dtype=torch.int, device=track_age.device)
        track_age_unmat = track_age[inactive_tracks] + 1
        new_track_age = torch.cat([det_age, track_age_unmat], 0)

        # build graph for new tracks
        track_adj_new = graph_util.bev_euclidean_distance_adj(new_track_boxes, new_track_class,
                                                              self.graph_truncation_dist)
        edge_index_track_new = graph_util.adj_to_edge_index(track_adj_new)

        self.track_state = {'features': new_track_feat,
                            'boxes': new_track_boxes,
                            'velo': new_track_velo,
                            'edge_index': edge_index_track_new,
                            'classes': new_track_class,
                            'gt_tracking_id': new_track_gt,
                            'age': new_track_age
                            }

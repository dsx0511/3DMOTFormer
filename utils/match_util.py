# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# The function greedy_assignment is taken from CenterPoint 
# (https://github.com/tianweiy/CenterPoint)
# Copyright (c) 2020-2021 Tianwei Yin and Xingyi Zhou. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)

def dets_tracks_matching(affinity, class_valid_mask, num_dets, num_tracks,
                         active_thresh=0.1, hungarian=True):
    threshold_mask = affinity > active_thresh
    valid_mask = torch.logical_and(class_valid_mask, threshold_mask)
    invalid_mask = torch.logical_not(valid_mask)

    cost = - affinity + 1e18 * invalid_mask
    cost[cost > 1e16] = 1e18
    # row_ind: index of tracks (last frames)
    # col_ind: index of detections (current frame)
    # cost[row_ind, col_ind] = c[i, j]
    if hungarian:
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    else:
        match_ind = greedy_assignment(cost.detach().cpu().numpy().transpose())
        row_ind = match_ind[:, 1]
        col_ind = match_ind[:, 0]

    unmatched_tracks = [t for t in range(num_tracks) if not (t in row_ind)]
    unmatched_dets = [d for d in range(num_dets) if not (d in col_ind)]

    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] > 1e16:
            unmatched_tracks.append(i)
            unmatched_dets.append(j)
        else:
            matches.append([i, j])
    
    assert len(unmatched_dets) + len(matches) == num_dets
    assert len(unmatched_tracks) + len(matches) == num_tracks
    matches = np.array(matches).reshape(-1, 2)
    
    return matches, unmatched_dets, unmatched_tracks

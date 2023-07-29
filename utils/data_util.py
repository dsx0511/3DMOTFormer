# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch
from torch_geometric.data import Data


def np_one_hot(in_arr, num_classes):
    out_arr = np.zeros((in_arr.size, num_classes), dtype=np.float64)
    out_arr[np.arange(in_arr.size), in_arr] = 1
    return out_arr

def torch_one_hot(in_arr, num_classes):
    in_arr = in_arr.to(torch.int64)
    out_arr = torch.nn.functional.one_hot(in_arr, num_classes=num_classes).float()
    return out_arr

def torch_k_hot(in_arr, num_classes):
    out_arr = torch.zeros(num_classes).float()
    out_arr[in_arr] = 1.0
    return out_arr

class BipartiteData(Data):
    def __init__(self, size_s=None, size_t=None, edge_index=None, edge_attr=None):
        super().__init__()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.size_s = size_s
        self.size_t = size_t

    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            return torch.tensor([[self.size_s], [self.size_t]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

NuScenesClasses = {
    'car' : 0,
    'pedestrian' : 1,
    'bicycle' : 2,
    'bus' : 3,
    'motorcycle' : 4,
    'trailer' : 5,
    'truck' : 6,
}
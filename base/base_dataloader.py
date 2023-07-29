# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Pytorch Template Project (https://github.com/victoresque/pytorch-template)
# Copyright (c) 2018 Victor Huang. All Rights Reserved.
# ------------------------------------------------------------------------

from torch_geometric.loader import DataLoader


class BaseDataLoader(DataLoader):
    
    def __init__(self, dataset, batch_size=1, pin_memory=True, shuffle=True, num_workers=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.dataset = dataset
        follow_batch = self.dataset.follow_batch if hasattr(self.dataset, 'follow_batch') else None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': True,
            'follow_batch': follow_batch
        }
        
        super().__init__(**self.init_kwargs)
    
    @property
    def sample_length(self):
        return self.dataset.sample_length
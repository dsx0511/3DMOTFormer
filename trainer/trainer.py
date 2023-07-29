# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Pytorch Template Project (https://github.com/victoresque/pytorch-template)
# Copyright (c) 2018 Victor Huang. All Rights Reserved.
# ------------------------------------------------------------------------

import json
import os
import copy

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils.unbatch import unbatch

from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils import match_util
import utils.graph_util as graph_util
from eval.tracker import Tracker, update_field, update_feature
from eval.nusc_eval import eval_nusc_tracking


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_dataset=None, eval_interval=1,
                 lr_scheduler=None, active_track_thresh=0.1, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_dataset = valid_dataset
        self.do_validation = self.valid_dataset is not None
        self.eval_interval = eval_interval
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['log_step']
        self.active_track_thresh = active_track_thresh
        self.max_age = self.config['trainer']['max_age']
        self.feature_update_weight = self.config['trainer']['feature_update_weight']
        self.hungarian = self.config['trainer']['hungarian_matching']
        self.graph_truncation_dist = self.config['graph_truncation_dist']
        self.nusc_path = self.config['trainer']['nusc_path']

        # A tracker class that manages tracklets and writes results during inference
        self.tracker = Tracker(max_age=self.max_age,
                               active_track_thresh=self.active_track_thresh,
                               feature_update_weight=self.feature_update_weight,
                               hungarian=self.hungarian,
                               graph_truncation_dist=self.graph_truncation_dist)
   
        train_loss_metrics = []
        for i in range(1, self.data_loader.sample_length):
            train_loss_metrics.append(f'loss/time_stamp_{i+1}')
        self.train_metrics = MetricTracker('loss/total', *train_loss_metrics, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', writer=None)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
    
    def _batch_update_tracks(self, batch_affinity, batch_inter_graph,
                                   batch_det_feat, batch_track_feat,
                                   batch_det_boxes, batch_track_boxes,
                                   batch_det_velo, batch_track_velo,
                                   batch_det_class, batch_track_class,
                                   batch_det_gt, batch_track_gt, batch_track_age,
                                   det_batch, track_batch
                                   ):
        '''
        Unbatch each batch element since the matching inside a graph is not parallelizeable.
        For each batch element, this function runs a bipartite matching and updates tracks using
        the matching results.
        A track instance is assigned with feature, box, velocity, category and matched object ID,
        and this function update these fields using different rules.
        '''
        inter_graph_list = batch_inter_graph.to_data_list()
        edge_index_inter_batch = batch_inter_graph.edge_index_batch
        affinity_list = unbatch(batch_affinity, edge_index_inter_batch)

        det_feat_list = unbatch(batch_det_feat, det_batch)
        track_feat_list = unbatch(batch_track_feat, track_batch)

        det_boxes_list = unbatch(batch_det_boxes, det_batch)
        track_boxes_list = unbatch(batch_track_boxes, track_batch)

        det_velo_list = unbatch(batch_det_velo, det_batch)
        track_velo_list = unbatch(batch_track_velo, track_batch)

        det_class_list = unbatch(batch_det_class, det_batch)
        track_class_list = unbatch(batch_track_class, track_batch)

        det_gt_list = unbatch(batch_det_gt, det_batch)
        track_gt_list = unbatch(batch_track_gt, track_batch)

        track_age_list = unbatch(batch_track_age, track_batch)

        data = []

        for (inter_graph, affinity, det_feat, trk_feat, det_boxes, trk_boxes, 
             det_velo, trk_velo, det_class, trk_class, det_gt, trk_gt, trk_age) in zip(
                inter_graph_list, affinity_list, det_feat_list, track_feat_list,
                det_boxes_list, track_boxes_list, det_velo_list, track_velo_list,
                det_class_list, track_class_list,
                det_gt_list, track_gt_list, track_age_list):

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
            
            match, unmat_det, unmat_trk = match_util.dets_tracks_matching(
                affinity_dense, adj, num_dets, num_tracks,
                active_thresh=self.active_track_thresh, hungarian=self.hungarian)

            inactive_trk = []
            for t in unmat_trk:
                if trk_age[t] < self.max_age:
                    inactive_trk.append(t)


            # Field for new tracks, the order follows [*matched track, *new_dets, *inactive_tracks]
            new_track_feat = update_feature(det_feat, trk_feat, match, unmat_det,
                                          inactive_trk, weight=self.feature_update_weight)
            new_track_boxes = update_field(det_boxes, trk_boxes, match, unmat_det,
                                           inactive_trk, update_with_dets=True)
            new_track_velo = update_field(det_velo, trk_velo, match, unmat_det,
                                          inactive_trk, update_with_dets=True)
            new_track_class = update_field(det_class, trk_class, match, unmat_det,
                                           inactive_trk, update_with_dets=True)
            new_track_gt = update_field(det_gt, trk_gt, match, unmat_det,
                                        inactive_trk, update_with_dets=True)

            det_age = torch.ones(len(match) + len(unmat_det), dtype=torch.int, device=trk_age.device)
            trk_age_unmat = trk_age[inactive_trk] + 1
            new_track_age = torch.cat([det_age, trk_age_unmat], 0)

            new_track_adj = graph_util.bev_euclidean_distance_adj(new_track_boxes, new_track_class, 
                                                                  self.graph_truncation_dist)
            new_track_edge_index = graph_util.adj_to_edge_index(new_track_adj)

            new_track_data = Data(x=new_track_feat,
                                  edge_index=new_track_edge_index,
                                  boxes=new_track_boxes,
                                  velo=new_track_velo,
                                  classes=new_track_class,
                                  tracking_id=new_track_gt,
                                  ages=new_track_age)
            data.append(new_track_data)
        
        data_batch = Batch.from_data_list(data)

        return data_batch


    def _train_mini_seq(self, data_seq):
        '''
        Training function for a clipped mini-sequence.
        It runs the model, updates the tracks and calculates losses frame-by-frame.

        :param data_seq: A list with torch_geometric.data.Data instances.
        :return losses: A dictionary which holds losses for each frame.
        '''
        losses = {}
        
        for i, data in enumerate(data_seq):
            if i == 0:
                # For first time stamp, initialize all detections as a track
                tracks = data.x
                track_boxes = data.det_box
                track_velo = data.det_velo
                edge_index_track = data.edge_index
                track_class = data.det_class
                track_batch = data.x_batch
                track_gt = data.tracking_id
                track_age = torch.ones_like(track_class, dtype=torch.int)
            else:
                dets = data.x
                det_boxes = data.det_box
                edge_index_det = data.edge_index
                det_class = data.det_class
                det_batch = data.x_batch
                det_gt = data.tracking_id
                velo_target = data.velo_target
                velo_mask = data.next_exist

                # Build inter graph between detections and tracks
                batched_inter_graph = graph_util.build_inter_graph(
                    det_boxes, track_boxes, det_class, track_class,
                    track_velo, track_age, det_batch, track_batch)
                edge_index_inter = batched_inter_graph.edge_index
                edge_attr_inter = batched_inter_graph.edge_attr

                # Forward pass
                affinity, track_feat, det_feat, pred_velo = self.model(
                    dets, tracks, edge_index_det, edge_index_track, edge_index_inter, edge_attr_inter)

                # loss
                target = self.criterion.generate_target(track_gt, det_gt, edge_index_inter)
                loss = self.criterion(affinity, target, pred_velo, velo_target, velo_mask)
                losses[f'loss/time_stamp_{i+1}'] = loss

                # update tracks
                if i != (len(data_seq) - 1):
                    affinity_sigmoid = torch.sigmoid(affinity[-1])
                    updated_track_data = self._batch_update_tracks(
                        affinity_sigmoid, batched_inter_graph, det_feat, track_feat,
                        det_boxes, track_boxes, pred_velo, track_velo, det_class, track_class,
                        det_gt, track_gt, track_age, det_batch, track_batch)

                    tracks = updated_track_data.x
                    track_boxes = updated_track_data.boxes
                    track_velo = updated_track_data.velo
                    track_class = updated_track_data.classes
                    edge_index_track = updated_track_data.edge_index
                    track_batch = updated_track_data.batch
                    track_gt = updated_track_data.tracking_id
                    track_age = updated_track_data.ages

        return losses
     
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data_seq in enumerate(self.data_loader):
            data_seq = [d.to(self.device) for d in data_seq]

            self.optimizer.zero_grad()
            loss_dict = self._train_mini_seq(data_seq)
            total_loss = torch.stack(list(loss_dict.values()), dim=0).sum()

            total_loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            log_str = f'Total loss: {total_loss}, '
            for k, v in loss_dict.items():
                self.train_metrics.update(k, v)
                log_str += f'{k}: {v}, '
            self.train_metrics.update('loss/total', total_loss)

            if (batch_idx + 1) % self.log_step == 0 or (batch_idx + 1) == self.len_epoch:
                self.logger.debug('Train Epoch: {} {} '.format(
                    epoch,
                    self._progress(batch_idx + 1)) + log_str)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            if epoch % self.eval_interval == 0:
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_'+ k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_seq(self, data_seq):
        '''
        Inference function for an entire sequence.
        It runs the model, updates the tracks and calculates losses frame-by-frame.

        :param data_seq: A list with torch_geometric.data.Data instances.
        :return annos: A dict with tracking results in nuscenes format.
        '''
        seq_length = len(data_seq)

        self.tracker.reset()
        annos = {}

        for i, (data, token) in enumerate(data_seq):
            data = data.to(self.device)

            if data.x.size(0) == 0:
                # TODO: output result here
                annos.update({token: []})
                continue
                
            if not self.tracker.is_initialized:
                self.tracker.init_tracks(data)
            else:
                dets = data.x
                det_boxes = data.det_box
                edge_index_det = data.edge_index
                det_class = data.det_class
                det_gt = data.tracking_id
                velo_target = data.velo_target
                velo_mask = data.next_exist

                det_batch = torch.zeros_like(det_class, dtype=torch.long, device=det_class.device)
                track_batch = torch.zeros_like(self.tracker.track_state['classes'],
                                               dtype=torch.long, device=det_class.device)

                inter_graph = graph_util.build_inter_graph(
                    det_boxes, self.tracker.track_state['boxes'],
                    det_class, self.tracker.track_state['classes'],
                    self.tracker.track_state['velo'], self.tracker.track_state['age'],
                    det_batch, track_batch).to_data_list()[0] # Batch size is 1 here
                edge_index_inter = inter_graph.edge_index
                edge_attr_inter = inter_graph.edge_attr

                affinity, track_feat, det_feat, pred_velo = self.model(
                    dets, self.tracker.track_state['features'],
                    edge_index_det, self.tracker.track_state['edge_index'],
                    edge_index_inter, edge_attr_inter)

                # loss
                target = self.criterion.generate_target(
                    self.tracker.track_state['gt_tracking_id'], det_gt, edge_index_inter)
                loss = self.criterion(affinity, target, pred_velo, velo_target, velo_mask)

                self.valid_metrics.update('loss', loss)

                # track step
                affinity = torch.sigmoid(affinity[-1])
                self.tracker.track_step(affinity, pred_velo, inter_graph, data, track_feat, det_feat)
        
            frame_annos = []
            unsorted_track_info = copy.deepcopy(self.tracker.track_info)
            sorted_track_info = sorted(unsorted_track_info,
                                       key=lambda d: d['tracking_score'],
                                       reverse=True)
            for item in sorted_track_info[:500]:
                score_factor = 1.0
                if item['active'] == 0:
                    score_factor = 0.1
                nusc_anno = {
                    "sample_token": token,
                    "translation": item['translation'].tolist(),
                    "size": item['size'].tolist(),
                    "rotation": item['rotation'].tolist(),
                    "velocity": item['velocity'].tolist(),
                    "tracking_id": str(item['tracking_id']),
                    "tracking_name": item['tracking_name'],
                    "tracking_score": float(item['tracking_score']) * score_factor,
                }
                frame_annos.append(nusc_anno)
            annos.update({token: frame_annos})

        return annos

    def _valid_epoch(self, epoch, val_outputs=None):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        nusc_annos = {
            "results": {},
            "meta": {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            }
        }
        self.writer.set_step(epoch, mode='val')

        print('Start validation...')
        with torch.no_grad():
            for i, data_seq in enumerate(self.valid_dataset):
                print(f'Inference on the {i+1}-th validtion sequence.')
                annos = self._valid_seq(data_seq)
                nusc_annos["results"].update(annos)
            
            if val_outputs is None:
                val_outputs = self.checkpoint_dir / 'val' / f'epoch_{epoch}'
            val_outputs.mkdir(parents=True, exist_ok=True)
            json_output = val_outputs / f'tracking_result_epoch_{epoch}.json'
            with open(json_output, "w") as f:
                json.dump(nusc_annos, f)
            
            metrics_summary = eval_nusc_tracking(json_output, 'val', val_outputs, self.nusc_path,
                                                 verbose=True,
                                                 num_vis=self.config['trainer']['num_vis'])

        mot_metrics = {
            "mot_metrics/amota": metrics_summary["amota"],
            "mot_metrics/amotp": metrics_summary["amotp"],
            "mot_metrics/motar": metrics_summary["motar"],
            "mot_metrics/mota": metrics_summary["mota"],
            "mot_metrics/motp": metrics_summary["motp"],
            "mot_metrics/recall": metrics_summary["recall"],

            "count_metrics/gt": metrics_summary["gt"],
            "count_metrics/mt": metrics_summary["mt"],
            "count_metrics/ml": metrics_summary["ml"],
            "count_metrics/tp": metrics_summary["tp"],
            "count_metrics/fp": metrics_summary["fp"],
            "count_metrics/fn": metrics_summary["fn"],
            "count_metrics/ids": metrics_summary["ids"],
            "count_metrics/frag": metrics_summary["frag"],
            "count_metrics/faf": metrics_summary["faf"],

            "time_metrics/tid": metrics_summary["tid"],
            "time_metrics/lgd": metrics_summary["lgd"],
        }

        for k, v in mot_metrics.items():
            self.writer.add_scalar(k, v)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

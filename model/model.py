# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.nn as nn
import copy

from base import BaseModel
from model.transformers import FFN, TransformerEncoderLayer, TransformerDecoderLayer

class STTransformerModel(BaseModel):
    def __init__(self, d_model, nhead, dropout, encoder_nlayers, decoder_nlayers, 
                       norm_first, cross_attn_value_gate, num_classes):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout_p = dropout
        self.encoder_nlayers = encoder_nlayers
        self.decoder_nlayers = decoder_nlayers
        self.norm_first = norm_first
        self.num_classes = num_classes
        self.edge_attr_cross_attn = True
        self.cross_attn_value_gate = cross_attn_value_gate

        self.embedding = nn.Linear(17, self.d_model)
        self.embedding_edge = nn.Linear(10, self.d_model)

        spatial_encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, self.dropout_p,
                                                        norm_first=self.norm_first)
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nhead, self.dropout_p,
                                                norm_first=self.norm_first,
                                                edge_attr_cross_attn=self.edge_attr_cross_attn,
                                                cross_attn_value_gate=self.cross_attn_value_gate)

        self.spatial_encoders = _get_clones(spatial_encoder_layer, self.encoder_nlayers)
        self.decoders = _get_clones(decoder_layer, self.decoder_nlayers)

        if self.norm_first:
            self.norm_final = nn.LayerNorm(self.d_model)
            self.norm_final_edge = nn.LayerNorm(self.d_model)

        self.affinity = FFN(self.d_model)
        self.velocity = FFN(self.d_model, 2)

    
    def forward(self, dets_in, tracks_in, edge_index_det, edge_index_track,
                edge_index_inter, edge_attr_inter=None):

        if tracks_in.size(1) != self.d_model:
            tracks = self.embedding(tracks_in)
        else:
            tracks = tracks_in
            
        dets = self.embedding(dets_in)
        edge_attr_inter = self.embedding_edge(edge_attr_inter)

        # Transformer encoder and decoder
        tracks = self._enc_block(tracks, edge_index_track)
        dets, edge_attr_inter, attn_weights = self._dec_block(
            tracks, dets, edge_index_inter, edge_index_det, edge_attr_inter)
        
        if self.norm_first:
            dets = self.norm_final(dets)
            edge_attr_inter = self.norm_final_edge(edge_attr_inter)

        affinity = [self.affinity(edge_attr_inter)]
        pred_velo = self.velocity(dets)

        return affinity, tracks, dets, pred_velo
    
    def _enc_block(self, x, edge_index):
        for spt_layer in self.spatial_encoders:
            x = spt_layer(x, edge_index)
        return x
    
    def _dec_block(self, src, tgt, edge_index_inter, edge_index_tgt, edge_attr_inter=None):
        attn_weights = []
        # tracks_out = []
        for layer in self.decoders:
            tgt, edge_attr_inter, attn = layer(src, tgt, edge_index_inter, edge_index_tgt, edge_attr_inter)
            # tracks_out.append(tracks)
            attn_weights.append(attn[1])
        return tgt, edge_attr_inter, attn_weights
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])





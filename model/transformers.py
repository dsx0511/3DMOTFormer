# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.nn as nn

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import TransformerConv

from base import BaseModel
from model.edge_augment_transformer_conv import EdgeAugmentTransformerConv

class FFN(BaseModel):

    def __init__(self, d_model, output_dim=1):
        super().__init__()

        self.d_model = d_model

        self.lin_1 = nn.Linear(self.d_model, self.d_model)
        self.lin_2 = nn.Linear(self.d_model, output_dim)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.lin_1(x)
        x = self.relu(x)
        x = self.lin_2(x)
        
        return x


class TransformerEncoderLayer(BaseModel):

    def __init__(self,
                 d_model,
                 heads=1,
                 dropout=0.0,
                 norm_first=False):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.dropout_p = dropout
        self.norm_first = norm_first

        self.head_channels = self.d_model // self.heads
        assert self.head_channels * self.heads == self.d_model, \
            'd_model must be dividable by heads'

        self.self_attn = TransformerConv(self.d_model, self.head_channels,
                                         heads=self.heads, dropout=self.dropout_p)

        self.lin1 = Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lin2 = Linear(self.d_model, self.d_model)
        self.activation = nn.ReLU()

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)

    def forward(self, x, edge_index):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), edge_index)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, edge_index))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x, edge_index):
        x = self.self_attn(x, edge_index)
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.lin2(self.dropout(self.activation(self.lin1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(BaseModel):

    def __init__(self,
                 d_model,
                 heads=1,
                 dropout=0.0,
                 norm_first=False,
                 apply_self_attn=True,
                 edge_attr_cross_attn=False,
                 cross_attn_value_gate=False):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.dropout_p = dropout
        self.norm_first = norm_first
        self.apply_self_attn = apply_self_attn
        self.edge_attr_cross_attn = edge_attr_cross_attn
        self.cross_attn_value_gate = cross_attn_value_gate

        self.head_channels = self.d_model // self.heads
        assert self.head_channels * self.heads == self.d_model, \
            'd_model must be dividable by heads'

        self.self_attn = TransformerConv(self.d_model, self.head_channels,
                                         heads=self.heads, dropout=self.dropout_p)

        self.edge_dim_cross_attn = self.d_model if self.edge_attr_cross_attn \
            else None
        self.cross_attn = EdgeAugmentTransformerConv(
            self.d_model, self.head_channels, gate=self.cross_attn_value_gate,
            heads=self.heads, dropout=self.dropout_p, edge_dim=self.edge_dim_cross_attn)

        self.lin1 = Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lin2 = Linear(self.d_model, self.d_model)
        self.activation = nn.ReLU()

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)

        self.lin_e1 = Linear(self.d_model, self.d_model)
        self.dropout_e = nn.Dropout(self.dropout_p)
        self.lin_e2 = Linear(self.d_model, self.d_model)
        self.activation_e = nn.ReLU()
        
        self.norm_e1 = nn.LayerNorm(self.d_model)
        self.norm_e2 = nn.LayerNorm(self.d_model)
        self.dropout_e1 = nn.Dropout(self.dropout_p)
        self.dropout_e2 = nn.Dropout(self.dropout_p)
    
    def forward(self, src, tgt, edge_index_cross, edge_index_tgt, 
                edge_attr_cra=None):
        x = tgt

        if self.norm_first:
            if self.apply_self_attn:
                x = x + self._sa_block(self.norm1(x), edge_index_tgt)
            x_delta, edge_attr_cra_delta, attn_weight = self._cra_block(
                src, self.norm2(x), edge_index_cross, self.norm_e1(edge_attr_cra))

            x = x + x_delta
            x = x + self._ff_block(self.norm3(x))

            edge_attr_cra = edge_attr_cra + edge_attr_cra_delta
            edge_attr_cra = self._ff_block_edge(self.norm_e2(edge_attr_cra))

        else:
            if self.apply_self_attn:
                x = self.norm1(x + self._sa_block(x, edge_index_tgt))
            x_delta, edge_attr_cra_delta, attn_weight = self._cra_block(
                src, x, edge_index_cross, edge_attr_cra)
            x = self.norm2(x + x_delta)
            x = self.norm3(x + self._ff_block(x))

            edge_attr_cra = self.norm_e1(edge_attr_cra + edge_attr_cra_delta)
            edge_attr_cra = self.norm_e2(edge_attr_cra + self._ff_block_edge(edge_attr_cra))
        return x, edge_attr_cra, attn_weight
    
    def _sa_block(self, x, edge_index):
        x = self.self_attn(x, edge_index)
        return self.dropout1(x)
    
    def _cra_block(self, src, tgt, edge_index, edge_attr=None):
        x = (src, tgt)
        if self.edge_attr_cross_attn:
            assert edge_attr is not None
        out, attn_weight = self.cross_attn(x, edge_index,
                                           edge_attr=edge_attr,
                                           return_attention_weights=True,
                                           return_edge_features=True)
        x, edge_feat = out
        return self.dropout2(x), edge_feat, attn_weight

    def _ff_block(self, x):
        x = self.lin2(self.dropout(self.activation(self.lin1(x))))
        return self.dropout3(x)

    def _ff_block_edge(self, e):
        e = self.lin_e2(self.dropout_e(self.activation_e(self.lin_e1(e))))
        return self.dropout_e2(e)

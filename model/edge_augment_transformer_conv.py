# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Pytorch Geometric (https://github.com/pyg-team/pytorch_geometric)
# Copyright (c) 2023 PyG Team <team@pyg.org>. All Rights Reserved.
# ------------------------------------------------------------------------

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax


class EdgeAugmentTransformerConv(MessagePassing):
    r"""
    Am implementation of the Edge-Augment Graph Transformer (https://arxiv.org/pdf/2108.03348.pdf)
    using pytorch geometric.
    Codes are adapted from the TransformerConv from `torch_geometric.nn.conv.transformer_conv`.
    """
    _alpha: OptTensor
    _edge_feat: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        gate: bool = False,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(EdgeAugmentTransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.gate = gate
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        self._edge_feat = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        self.lin_edge_attn = Linear(edge_dim, heads)
        self.lin_edge = Linear(heads, edge_dim)
        if self.gate:
            self.lin_edge_gate = Linear(edge_dim, heads * out_channels, bias=False)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge_attn.reset_parameters()
        self.lin_edge.reset_parameters()
        if self.gate:
            self.lin_edge_gate.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None,
                return_edge_features=False):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        edge_attn = self.lin_edge_attn(edge_attr)
        edge_gate = None
        if self.gate:
            edge_gate = self.lin_edge_gate(edge_attr).view(-1, H, C)
            edge_gate = torch.sigmoid(edge_gate)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attn=edge_attn, edge_gate=edge_gate, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_feat = self._edge_feat
        self._edge_feat = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r
        
        edge_feat = self.lin_edge(edge_feat)
        
        out = (out, edge_feat)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attn: OptTensor, edge_gate: OptTensor, 
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        # if self.lin_edge is not None:
        #     assert edge_attr is not None
        #     edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
        #                                               self.out_channels)
        #     key_j += edge_attr
        
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha += edge_attn
        self._edge_feat = alpha

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_gate is not None:
            value_j *= edge_gate

        out = value_j
        # if edge_attr is not None:
        #     out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

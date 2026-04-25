# -*- coding: utf-8 -*-
"""
- GraphEncoderLayer / GraphTransformerWithEdgeBias: encoder
- FiLMNoiseHead: modulates graph representations with observation features
- QIONFiLMModel: overall model (encoder + FiLM head)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
from torch import nn, Tensor


try:
    from fairseq.modules import LayerNorm
    from fairseq.modules.fairseq_dropout import FairseqDropout
except Exception:
    LayerNorm = nn.LayerNorm
    class FairseqDropout(nn.Dropout):
        def __init__(self, p: float, module_name: Optional[str] = None):
            super().__init__(p)

from multihead_attn import EdgeBiasMLP, MultiheadSelfAttentionEdgeBias


def _get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ======= Data container (aligned with _GraphSample in trainer) =======
@dataclass
class GraphSample:
    x: Tensor           # [N, D_node]
    edge_index: Tensor  # [2, E]
    edge_attr: Tensor   # [E, D_edge]


# ======= Graph Encoder =======
class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        ffn_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation: str = "gelu",
        pre_layernorm: bool = True,
    ):
        super().__init__()
        self.pre_layernorm = pre_layernorm
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttentionEdgeBias(
            embed_dim=embed_dim, num_heads=num_heads,
            attn_dropout=attn_dropout, proj_dropout=dropout
        )
        self.ln2 = LayerNorm(embed_dim)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            act,
            FairseqDropout(dropout, module_name=self.__class__.__name__),
            nn.Linear(ffn_dim, embed_dim),
            FairseqDropout(dropout, module_name=self.__class__.__name__),
        )

    def forward(
        self,
        x_TBC: Tensor,
        key_padding_mask: Optional[Tensor],
        edge_bias: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        y = self.ln1(x_TBC) if self.pre_layernorm else x_TBC
        y, _ = self.attn(y, key_padding_mask=key_padding_mask, edge_bias=edge_bias, attn_mask=attn_mask)
        x_TBC = x_TBC + y
        x_TBC = self.ln1(x_TBC) if not self.pre_layernorm else x_TBC

        y = self.ln2(x_TBC) if self.pre_layernorm else x_TBC
        y = self.ffn(y)
        x_TBC = x_TBC + y
        x_TBC = self.ln2(x_TBC) if not self.pre_layernorm else x_TBC
        return x_TBC


# ======= Graph Transformer (with edge bias) =======
class GraphTransformerWithEdgeBias(nn.Module):
    def __init__(
        self,
        d_node: int = 61,
        d_edge: int = 5,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        pre_layernorm: bool = True,
        restrict_to_edges: bool = False,     # True: add -1e9 to non-edge positions
        share_edge_bias_across_heads: bool = False,
        add_cls_token: bool = False,
        use_edge_bias: bool = True,
    ):
        super().__init__()
        self.embed = nn.Linear(d_node, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)

        self.layers = nn.ModuleList([
            GraphEncoderLayer(
                embed_dim=embed_dim, ffn_dim=ffn_dim, num_heads=num_heads,
                dropout=dropout, attn_dropout=attn_dropout, pre_layernorm=pre_layernorm
            ) for _ in range(num_layers)
        ])
        self.edge_bias_mlp = EdgeBiasMLP(d_edge, num_heads, share_across_heads=share_edge_bias_across_heads)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.restrict_to_edges = restrict_to_edges
        self.add_cls = add_cls_token
        if self.add_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.use_edge_bias = use_edge_bias

    @staticmethod
    def _pad_pack(graphs: List[GraphSample], device=None):
        """
        Pad graphs with different N to shape [T=maxN, B, D], and return the pad mask.
        """
        B = len(graphs)
        Ns = [int(g.x.size(0)) for g in graphs]
        D = int(graphs[0].x.size(1))
        T = max(Ns)

        x_pad = torch.zeros(T, B, D, device=device, dtype=graphs[0].x.dtype)
        pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        for b, g in enumerate(graphs):
            N = g.x.size(0)
            x_pad[:N, b, :] = g.x.to(device)
            pad_mask[b, :N] = False
        return x_pad, pad_mask, Ns

    def _build_edge_bias_and_mask(self, graphs: List[GraphSample], Ns: List[int], device):
        """
        Construct:
            - edge_bias: [B, H, T, T]
                Non-edge positions are set to 0.
            - attn_mask: None or [B, T, T]
                When restrict_to_edges=True, non-edge positions are masked with -1e9.
        """
        B, H, T = len(graphs), self.num_heads, max(Ns)
        edge_bias = torch.zeros(B, H, T, T, device=device)
        attn_mask = None
        if self.restrict_to_edges:
            attn_mask = torch.full((B, T, T), 0.0, device=device)

        for b, g in enumerate(graphs):
            N = int(g.x.size(0))
            eb = self.edge_bias_mlp.dense_bias(
                edge_index=g.edge_index.to(device),
                edge_attr=g.edge_attr.to(device),
                num_nodes=N, device=device
            )  # [H,N,N]
            edge_bias[b, :, :N, :N] = eb

            if self.restrict_to_edges:
                mask = torch.full((N, N), float("-1e9"), device=device)
                if g.edge_index.numel() > 0:
                    u, v = g.edge_index.long().to(device)
                    mask[u, v] = 0.0
                idx = torch.arange(N, device=device)
                mask[idx, idx] = 0.0
                attn_mask[b, :N, :N] = mask
        return edge_bias, attn_mask

    def forward(self, graphs: List[GraphSample]) -> Tensor:
        device = graphs[0].x.device
        x_TBC, pad_mask, Ns = self._pad_pack(graphs, device=device)

        x = self.embed(x_TBC)
        x = self.pos_ln(x)

        if self.add_cls:
            B = x.size(1)
            cls_tok = self.cls.expand(-1, B, -1)  # [1,B,C]
            x = torch.cat([cls_tok, x], dim=0)
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
            pad_mask = torch.cat([cls_pad, pad_mask], dim=1)
            Ns = [n + 1 for n in Ns]

        edge_bias, attn_mask = self._build_edge_bias_and_mask(graphs, Ns, device=device)
        if not self.use_edge_bias:
            edge_bias = None

        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask, edge_bias=edge_bias, attn_mask=attn_mask)

        # CLS or mask-mean
        if self.add_cls:
            h_graph = x[0]                        # [B,C]
        else:
            valid = (~pad_mask).float()           # [B,T]
            x_BTC = x.permute(1, 0, 2)           # [B,T,C]
            h_graph = (x_BTC * valid.unsqueeze(-1)).sum(dim=1) / (valid.sum(dim=1, keepdim=True) + 1e-9)
        return h_graph                            # [B, embed_dim]


# ======= FiLM Head: modulate x_obs onto h_graph =======
class FiLMNoiseHead(nn.Module):
    """
        x_obs → generate (gamma, beta) to modulate h_graph; includes a skip connection 
        to strengthen the influence of x_obs.
        Default obs_dim=4: [POS, POF, logit(POS), 1/sqrt(shots)]
    """
    def __init__(
        self,
        h_dim: int,
        obs_dim: int = 4,
        hidden: int = 256,
        direct_branch: bool = True,
        use_film: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.h_norm = nn.LayerNorm(h_dim)
        self.film = nn.Linear(obs_dim, 2 * h_dim)        # -> [γ|β]
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.direct_branch = direct_branch
        if direct_branch:
            self.obs_mlp = nn.Sequential(
                nn.Linear(obs_dim, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1)
            )
        self.use_film = use_film

    def forward(self, h_graph: Tensor, x_obs: Tensor) -> Tensor:
        """
        h_graph: [B, H]
        x_obs  : [B, D_obs]
        return : [B]
        """
        h = self.h_norm(h_graph)
        
        if self.use_film:
            gamma, beta = self.film(x_obs).chunk(2, dim=-1)  # [B,H], [B,H]
            h_tilde = (1 + gamma) * h + beta
        else:
            h_tilde = h 

        # basic scalar
        y = self.mlp(h_tilde).squeeze(-1)

        if self.direct_branch:
            y += self.obs_mlp(x_obs).squeeze(-1)
        
        return y


# ======= Overall model (Encoder + FiLM Head) =======
class QIONFiLMModel(nn.Module):
    """
        - The trainer will call:
            encoder(graphs_unique) -> [B_graph, H]
            head(h_all, x_obs_all) -> [B_state]

        - Hyperparameters are retrieved from configs.model.
        Explicit arguments (function parameters below) take priority,
        followed by values in configs.model, and finally default values.
    """
    def __init__(
        self,
        args,                                      
        d_node: Optional[int] = None,
        d_edge: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        obs_dim: Optional[int] = None,
        head_hidden: Optional[int] = None,
        restrict_to_edges: Optional[bool] = None,
        add_cls_token: Optional[bool] = None,
        share_edge_bias_across_heads: Optional[bool] = None,
        dropout: Optional[float] = None,
        attn_dropout: Optional[float] = None,
        pre_layernorm: Optional[bool] = None,
        use_edge_bias: Optional[bool] = None,
        use_film: Optional[bool] = None,
        direct_branch: Optional[bool] = None,
    ):
        super().__init__()
        self.args = args

        d_node = d_node if d_node is not None else _get(args, "d_node", None)
        d_edge = d_edge if d_edge is not None else _get(args, "d_edge", 5)

        embed_dim   = embed_dim   if embed_dim   is not None else _get(args, "embed_dim", 256)
        num_heads   = num_heads   if num_heads   is not None else _get(args, "num_heads", 8)
        num_layers  = num_layers  if num_layers  is not None else _get(args, "num_layers", 4)
        ffn_dim     = ffn_dim     if ffn_dim     is not None else _get(args, "ffn_dim", 512)
        obs_dim     = obs_dim     if obs_dim     is not None else _get(args, "obs_dim", 4)
        head_hidden = head_hidden if head_hidden is not None else _get(args, "head_hidden", 256)

        restrict_to_edges = restrict_to_edges if restrict_to_edges is not None else _get(args, "restrict_to_edges", False)
        add_cls_token     = add_cls_token     if add_cls_token     is not None else _get(args, "add_cls_token", False)
        share_edge_bias_across_heads = (
            share_edge_bias_across_heads if share_edge_bias_across_heads is not None
            else _get(args, "share_edge_bias_across_heads", False)
        )
        dropout       = dropout      if dropout      is not None else _get(args, "dropout", 0.1)
        attn_dropout  = attn_dropout if attn_dropout is not None else _get(args, "attn_dropout", 0.1)
        pre_layernorm = pre_layernorm if pre_layernorm is not None else _get(args, "pre_layernorm", True)
        use_edge_bias = use_edge_bias if use_edge_bias is not None else _get(args, "use_edge_bias", True)
        use_film      = use_film if use_film is not None else _get(args, "use_film", True)
        direct_branch = direct_branch if direct_branch is not None else _get(args, "direct_branch", True)



        if d_node is None or d_edge is None:
            raise ValueError(
                "d_node/d_edge not provided; please configure them in configs.model, or override via command line: model.d_node=... model.d_edge=..."
            )

        self.encoder = GraphTransformerWithEdgeBias(
            d_node=d_node, d_edge=d_edge,
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ffn_dim=ffn_dim,
            dropout=dropout, attn_dropout=attn_dropout, pre_layernorm=pre_layernorm,
            restrict_to_edges=restrict_to_edges, add_cls_token=add_cls_token,
            share_edge_bias_across_heads=share_edge_bias_across_heads, use_edge_bias=use_edge_bias,
        )
        self.head = FiLMNoiseHead(h_dim=embed_dim, obs_dim=obs_dim, hidden=head_hidden, direct_branch=True)

    # Convenient for directly converting POS/shots into x_obs elsewhere
    @staticmethod
    def make_x_obs(pos: Tensor, shots: Tensor, use_pof=True, use_logit=True, use_inv_sqrt_shots=True, eps=1e-6):
        """
        pos:   [B] (0,1)
        shots: [B] (positive)
        return:  [B, D_obs]
        """
        pos = pos.clamp(eps, 1 - eps)
        feats = [pos]
        if use_pof:
            feats.append(1.0 - pos)
        if use_logit:
            feats.append(torch.log(pos / (1.0 - pos)))
        if use_inv_sqrt_shots:
            feats.append(1.0 / torch.sqrt(shots.float().clamp_min(1.0)))
        return torch.stack(feats, dim=-1)

    def forward(self, graphs: List[GraphSample], x_obs: Tensor) -> Tensor:
        h = self.encoder(graphs)              # [B, H]
        p_hat = self.head(h, x_obs)           # [B]
        return p_hat

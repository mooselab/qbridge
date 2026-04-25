# -*- coding: utf-8 -*-
"""
- EdgeBiasMLP: maps per-edge features into additive attention biases for each head
- MultiheadSelfAttentionEdgeBias: standard self-attention with edge_bias added to QK^T logits
"""
from typing import Optional, Tuple

import math
import torch
from torch import nn, Tensor

try:
    from fairseq.modules.fairseq_dropout import FairseqDropout
except Exception:
    class FairseqDropout(nn.Dropout):
        def __init__(self, p: float, module_name: Optional[str] = None):
            super().__init__(p)


class EdgeBiasMLP(nn.Module):
    """
    Project edge features edge_attr[E, D_edge] into per-head biases,
    and write them into the dense bias tensor [H, N, N].
    """
    def __init__(self, d_edge: int, num_heads: int, share_across_heads: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.share = share_across_heads
        # If share=True, use one weight for all heads; otherwise generate one scalar per head
        self.proj = nn.Linear(d_edge, 1 if share_across_heads else num_heads)

    @torch.no_grad()
    def dense_bias(self, edge_index: Tensor, edge_attr: Tensor, num_nodes: int, device=None) -> Tensor:
        """
        edge_index: [2, E]  (values in range 0..N-1)
        edge_attr : [E, D_edge]
        Return: [H, N, N], with non-edge positions set to 0
        """
        if device is None:
            device = edge_attr.device
        H = self.num_heads
        bias = torch.zeros(H, num_nodes, num_nodes, device=device)

        if edge_index.numel() == 0:
            return bias

        scores = self.proj(edge_attr)  # [E,1] or [E,H]
        u, v = edge_index.long()
        if self.share:
            s = scores.squeeze(-1)     # [E]
            for h in range(H):
                bias[h, u, v] = s
        else:
            # scores: [E, H] -> [H, E] -> scatter to [H, N, N]
            bias[:, u, v] = scores.transpose(0, 1)
        return bias


class MultiheadSelfAttentionEdgeBias(nn.Module):
    """
    Self-attention only; add edge_bias to the logits before applying softmax.

    Input/Output shapes:
    - x_TBC: [T, B, C]
    - key_padding_mask: [B, T]  (True = padding)
    - edge_bias: [B, H, T, T]  (pass None if not provided)
    - attn_mask: [T, T] or [B, T, T]  (use -1e9 for masking)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = FairseqDropout(attn_dropout, module_name=self.__class__.__name__)
        self.proj_drop = FairseqDropout(proj_dropout, module_name=self.__class__.__name__)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.proj.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x_TBC: Tensor,
        key_padding_mask: Optional[Tensor],
        edge_bias: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        T, B, C = x_TBC.shape

        qkv = self.qkv(x_TBC)      # [T,B,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            t = t.contiguous().view(T, B, self.num_heads, C // self.num_heads)
            return t.permute(1, 2, 0, 3).reshape(B * self.num_heads, T, C // self.num_heads)

        q = split_heads(q) * self.scaling
        k = split_heads(k)
        v = split_heads(v)

        logits = torch.bmm(q, k.transpose(1, 2))  # [B*H, T, T]

        if edge_bias is not None:
            # edge_bias: [B, H, T, T] -> [B*H, T, T]
            logits = logits + edge_bias.view(B * self.num_heads, T, T)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                logits = logits + attn_mask.unsqueeze(0)
            else:  # [B,T,T]
                logits = logits + attn_mask.repeat_interleave(self.num_heads, dim=0)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).expand(B, T, T)     # [B,T,T]
            mask = mask.repeat_interleave(self.num_heads, dim=0)     # [B*H,T,T]
            logits = logits.masked_fill(mask, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        Dh = C // self.num_heads
        out = torch.bmm(attn, v)                                              # [B*H, T, Dh]
        out = out.view(B, self.num_heads, T, Dh).permute(2, 0, 1, 3).contiguous()  # [T, B, H, Dh]
        out = out.view(T, B, C)                                               # combine H*Dh -> C
        out = self.proj(out)
        out = self.proj_drop(out)

        if need_weights:
            return out, attn.view(B, self.num_heads, T, T)
        return out, None

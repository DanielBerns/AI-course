"""
Transformer Block & Multi-Head Self-Attention Implemented from Scratch in PyTorch.

Implements Scaled Dot-Product Attention, Causal Masking, Multi-Head Attention projection,
Sinusoidal Positional Encodings, and Transformer Layer blocks.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention with optional Causal Masking."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q, k, v shape: (batch_size, num_heads, seq_len, head_dim)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention Layer."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block with Layer Normalization & Residual Skips."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Architecture
        x_norm = self.norm1(x)
        x = x + self.mha(x_norm, mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


def get_causal_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    """Generate lower-triangular causal attention mask for autoregressive models."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    return mask  # Shape: (1, 1, seq_len, seq_len)

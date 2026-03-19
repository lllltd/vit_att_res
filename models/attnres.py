"""
Block Attention Residuals
Based on: Attention Residuals (arXiv: 2603.15031, Kimi Team 2026)
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm for key normalization in AttnRes."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class BlockAttnRes(nn.Module):
    """
    Single depth-attention operation: softmax attention over blocks + partial.
    
    Args:
        dim: hidden dimension (must match last dim of input tensors)
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))   # zero-init (paper requirement)
        self.norm = RMSNorm(dim)

    def forward(self, blocks, partial_block):
        """
        Args:
            blocks: list of tensors [..., D] — completed block representations
            partial_block: tensor [..., D] — current block's partial accumulation
        Returns:
            tensor [..., D] — weighted aggregation
        """
        V = torch.stack(blocks + [partial_block], dim=0)    # [N+1, ...]
        K = self.norm(V)                                      # [N+1, ...]
        logits = (self.query * K).sum(dim=-1)                 # [N+1, ...(no D)]
        weights = logits.softmax(dim=0).unsqueeze(-1)         # [N+1, ..., 1]
        return (weights * V).sum(dim=0)                       # [..., D]

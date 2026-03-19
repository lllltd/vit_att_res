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
    Pseudo-query is zero-initialized (paper requirement).
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim)

    def forward(self, blocks, partial_block):
        """
        Args:
            blocks: list of tensors [..., D]
            partial_block: tensor [..., D]
        Returns:
            tensor [..., D] — weighted aggregation
        """
        V = torch.stack(blocks + [partial_block], dim=0)
        K = self.norm(V)
        logits = (self.query * K).sum(dim=-1)
        weights = logits.softmax(dim=0).unsqueeze(-1)
        return (weights * V).sum(dim=0)

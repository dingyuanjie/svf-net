import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SVFAttention(nn.Module):
    """Multi-head self-attention augmented with a structural potential field."""

    def __init__(self, dim: int, heads: int = 8, lambda_svf: float = 0.1,
                 max_len: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim, self.heads, self.max_len = dim, heads, max_len
        self.head_dim = dim // heads
        self.lambda_svf = lambda_svf
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # These are registered parameters, so they move correctly with .to(device).
        self.theta = nn.Parameter(torch.empty(max_len))
        self.radius = nn.Parameter(torch.linspace(0.0, 1.0, max_len))
        nn.init.uniform_(self.theta, -torch.pi, torch.pi)

    def structural_field(self, length: int) -> Tensor:
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        theta, radius = self.theta[:length], self.radius[:length]
        angle = torch.cos(theta[:, None] - theta[None, :])
        scale = torch.exp(-torch.abs(radius[:, None] - radius[None, :]))
        return 0.5 * (angle + scale)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + self.lambda_svf * self.structural_field(length)
        if causal:
            mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, length, channels)
        return self.out(output)

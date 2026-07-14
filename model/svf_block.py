from torch import Tensor, nn

from .svf_attention import SVFAttention


class SVFBlock(nn.Module):
    def __init__(self, dim: int, heads: int, max_len: int, lambda_svf: float = 0.1,
                 dropout: float = 0.0, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SVFAttention(dim, heads, lambda_svf, max_len, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        return x + self.ffn(self.norm2(x))

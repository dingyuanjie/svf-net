import torch
from torch import Tensor, nn

from .svf_block import SVFBlock


class SVFTransformer(nn.Module):
    """SVF-Transformer v0.1 language-model backbone."""

    def __init__(self, vocab: int, dim: int = 512, depth: int = 6, heads: int = 8,
                 max_len: int = 2048, lambda_svf: float = 0.1, dropout: float = 0.0,
                 field_mode: str = "svf") -> None:
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.layers = nn.ModuleList([
            SVFBlock(dim, heads, max_len, lambda_svf, dropout=dropout,
                     field_mode=field_mode) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

    def forward(self, tokens: Tensor, causal: bool = True) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        _, length = tokens.shape
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        hidden = self.token_emb(tokens) + self.pos_emb[:, :length]
        for layer in self.layers:
            hidden = layer(hidden, causal=causal)
        return self.head(self.norm(hidden))

from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 50_000
    dim: int = 512
    depth: int = 6
    heads: int = 8
    max_len: int = 512
    lambda_svf: float = 0.1
    dropout: float = 0.0
    batch_size: int = 8
    steps: int = 1_000
    learning_rate: float = 3e-4

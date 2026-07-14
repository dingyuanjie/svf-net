import argparse

import torch
from torch.nn import functional as F

from config import Config
from model import SVFTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SVF-Transformer v0.1 toy language model")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = Config()
    steps = args.steps or cfg.steps
    device = torch.device(args.device)
    model = SVFTransformer(cfg.vocab_size, cfg.dim, cfg.depth, cfg.heads,
                           cfg.max_len, cfg.lambda_svf, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model.train()
    for step in range(steps):
        tokens = torch.randint(cfg.vocab_size, (cfg.batch_size, cfg.max_len + 1), device=device)
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step:6d} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

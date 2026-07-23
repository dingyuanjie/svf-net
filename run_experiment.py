"""Run the complete SVF-Net v0.1-alpha ablation in one command."""

import argparse
import csv
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from config import Config
from model import SVFTransformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(cfg: Config, device: torch.device, generator: torch.Generator,
               task: str, batch_size: int | None = None) -> torch.Tensor:
    batch_size = batch_size or cfg.batch_size
    if task == "random":
        return torch.randint(cfg.vocab_size, (batch_size, cfg.max_len + 1),
                             device=device, generator=generator)
    if task == "copy":
        # First half is random; the second half repeats it. The model must
        # retrieve information from far back in the context.
        half = cfg.max_len // 2
        prefix = torch.randint(cfg.vocab_size, (batch_size, half),
                               device=device, generator=generator)
        repeat = prefix.repeat(1, (cfg.max_len + 1 - half + half - 1) // half)
        return torch.cat((prefix, repeat[:, :cfg.max_len + 1 - half]), dim=1)
    raise ValueError(f"unknown task: {task}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, cfg: Config, device: torch.device,
             task: str, seed: int, batches: int = 10) -> dict:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count, correct, copy_count = 0.0, 0, 0, 0
    for _ in range(batches):
        tokens = make_batch(cfg, device, generator, task)
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
        total += float(loss) * tokens[:, 1:].numel()
        count += tokens[:, 1:].numel()
        if task == "copy":
            start = cfg.max_len // 2 - 1
            predictions = logits[:, start:].argmax(dim=-1)
            targets = tokens[:, 1 + start:]
            correct += int((predictions == targets).sum())
            copy_count += targets.numel()
    model.train()
    result = {"loss": total / count}
    if task == "copy":
        result["copy_accuracy"] = correct / copy_count
    return result


def train_one(mode: str, cfg: Config, steps: int, device: torch.device, seed: int,
              log_path: Path, task: str, eval_every: int) -> dict:
    # Reset all RNGs so each model sees the same synthetic batches and initialization policy.
    seed_everything(seed)
    model = SVFTransformer(cfg.vocab_size, cfg.dim, cfg.depth, cfg.heads,
                           cfg.max_len, cfg.lambda_svf, cfg.dropout, mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    parameter_count = sum(p.numel() for p in model.parameters())
    model.train()
    losses = []
    started = time.perf_counter()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "val_loss", "copy_accuracy"])
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        for step in range(steps):
            tokens = make_batch(cfg, device, generator, task)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            val_loss = ""
            copy_accuracy = ""
            if step == 0 or (step + 1) % eval_every == 0 or step == steps - 1:
                validation = evaluate(model, cfg, device, task, seed + 2000)
                val_loss = validation["loss"]
                copy_accuracy = validation.get("copy_accuracy", "")
            writer.writerow([step, value, val_loss, copy_accuracy])
            if step == 0 or (step + 1) % max(1, steps // 10) == 0:
                suffix = f" val_loss={val_loss:.4f}" if val_loss != "" else ""
                if copy_accuracy != "":
                    suffix += f" copy_acc={copy_accuracy:.4f}"
                print(f"[{mode:8s}] step={step + 1:5d}/{steps} loss={value:.4f}{suffix}")
    return {
        "mode": mode,
        "parameters": parameter_count,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "final_validation": evaluate(model, cfg, device, task, seed + 2000),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all SVF-Net v0.1-alpha ablations")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/ablation")
    parser.add_argument("--task", choices=("copy", "random"), default="copy")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--small-copy", action="store_true",
                        help="use a smaller diagnostic copy benchmark")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    cfg = Config()
    if args.small_copy:
        cfg = replace(cfg, vocab_size=32, dim=128, depth=4, heads=4, max_len=128, batch_size=16)
    cfg = replace(cfg, steps=args.steps)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = []
    print(f"SVF-Net v0.1-alpha | task={args.task} | device={device} | steps={args.steps} | seed={args.seed}")
    for mode in ("baseline", "svf", "random"):
        results.append(train_one(mode, cfg, args.steps, device, args.seed,
                                 output / f"{mode}.csv", args.task, args.eval_every))

    summary = {
        "version": "v0.1-alpha",
        "config": vars(cfg),
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "task": args.task,
        "results": results,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n完成。结果文件：")
    print(f"  {summary_path}")
    for item in results:
        print(f"  {item['mode']:8s} final_loss={item['final_loss']:.4f} log={item['log']}")


if __name__ == "__main__":
    main()

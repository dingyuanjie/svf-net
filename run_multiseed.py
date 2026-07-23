"""Run the copy ablation across multiple random seeds and aggregate statistics."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from run_experiment import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SVF-Net multi-seed ablation")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/multiseed")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("--seeds must contain at least one integer")

    cfg = replace(Config(), vocab_size=32, dim=128, depth=4, heads=4,
                  max_len=128, batch_size=16, steps=args.steps)
    root = Path(args.output)
    device = torch.device(args.device)
    all_results = []
    for seed in seeds:
        seed_dir = root / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== seed {seed} =====")
        for mode in ("baseline", "svf", "random"):
            result = train_one(mode, cfg, args.steps, device, seed,
                               seed_dir / f"{mode}.csv", "copy", max(1, args.steps // 20))
            result["seed"] = seed
            all_results.append(result)

    aggregate = {}
    for mode in ("baseline", "svf", "random"):
        values = [item for item in all_results if item["mode"] == mode]
        losses = [item["final_validation"]["loss"] for item in values]
        accuracies = [item["final_validation"]["copy_accuracy"] for item in values]
        mean_loss = sum(losses) / len(losses)
        mean_accuracy = sum(accuracies) / len(accuracies)
        aggregate[mode] = {
            "seeds": len(values),
            "mean_val_loss": mean_loss,
            "std_val_loss": (sum((x - mean_loss) ** 2 for x in losses) / len(losses)) ** 0.5,
            "mean_copy_accuracy": mean_accuracy,
            "std_copy_accuracy": (sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5,
            "runs": values,
        }

    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "v0.1-alpha-multiseed",
        "task": "copy",
        "config": vars(cfg),
        "device": str(device),
        "seeds": seeds,
        "aggregate": aggregate,
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n多 seed 实验完成：{path}")
    for mode, result in aggregate.items():
        print(f"{mode:8s} val_loss={result['mean_val_loss']:.5f} ± {result['std_val_loss']:.5f} | "
              f"copy_acc={result['mean_copy_accuracy']:.4f} ± {result['std_copy_accuracy']:.4f}")


if __name__ == "__main__":
    main()

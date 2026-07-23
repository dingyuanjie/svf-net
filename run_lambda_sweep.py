"""Sweep the SVF structural-field strength on the diagnostic copy task."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from run_experiment import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed lambda_svf sweep")
    parser.add_argument("--lambdas", default="0,0.01,0.05,0.1,0.2,0.5")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/lambda-sweep")
    args = parser.parse_args()
    lambdas = [float(value.strip()) for value in args.lambdas.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not lambdas or not seeds:
        parser.error("--lambdas and --seeds must not be empty")

    cfg_base = replace(Config(), vocab_size=32, dim=128, depth=4, heads=4,
                       max_len=128, batch_size=16, steps=args.steps)
    root, device = Path(args.output), torch.device(args.device)
    all_results = []
    for value in lambdas:
        print(f"\n========== lambda_svf={value} ==========")
        for seed in seeds:
            cfg = replace(cfg_base, lambda_svf=value)
            folder = root / f"lambda-{value:g}" / f"seed-{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            result = train_one("svf", cfg, args.steps, device, seed,
                               folder / "svf.csv", "copy", max(1, args.steps // 20))
            result.update({"lambda_svf": value, "seed": seed})
            all_results.append(result)

    aggregate = []
    for value in lambdas:
        runs = [r for r in all_results if r["lambda_svf"] == value]
        losses = [r["final_validation"]["loss"] for r in runs]
        accuracies = [r["final_validation"]["copy_accuracy"] for r in runs]
        mean_loss = sum(losses) / len(losses)
        mean_accuracy = sum(accuracies) / len(accuracies)
        aggregate.append({
            "lambda_svf": value,
            "mean_val_loss": mean_loss,
            "std_val_loss": (sum((x - mean_loss) ** 2 for x in losses) / len(losses)) ** 0.5,
            "mean_copy_accuracy": mean_accuracy,
            "std_copy_accuracy": (sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5,
            "runs": runs,
        })

    root.mkdir(parents=True, exist_ok=True)
    summary = {"version": "v0.1-alpha-lambda-sweep", "task": "copy",
               "config": vars(cfg_base), "device": str(device), "seeds": seeds,
               "lambdas": lambdas, "aggregate": aggregate}
    path = root / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n扫描完成：{path}")
    print("lambda\tval_loss\tcopy_accuracy")
    for item in aggregate:
        print(f"{item['lambda_svf']:g}\t{item['mean_val_loss']:.5f} ± {item['std_val_loss']:.5f}\t"
              f"{item['mean_copy_accuracy']:.4f} ± {item['std_copy_accuracy']:.4f}")


if __name__ == "__main__":
    main()

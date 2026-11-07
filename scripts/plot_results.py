"""Plot batch-level experiment outputs produced by ``afd.run``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot adaptive fraud experiment results")
    parser.add_argument("--results", required=True, help="Directory containing batch_metrics.csv")
    parser.add_argument("--output", default=None, help="PNG path; defaults to results/metrics.png")
    args = parser.parse_args()

    results_dir = Path(args.results)
    batch_path = results_dir / "batch_metrics.csv"
    if not batch_path.exists():
        raise FileNotFoundError(f"Missing batch metrics: {batch_path}")
    frame = pd.read_csv(batch_path)
    if frame.empty:
        raise ValueError("batch_metrics.csv contains no rows")

    output = Path(args.output) if args.output else results_dir / "metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for seed, group in frame.groupby("seed"):
        axes[0].plot(
            group["batch_index"],
            group["pr_auc_rolling"],
            marker="o",
            linewidth=1,
            label=f"seed={seed}",
        )
        axes[1].plot(
            group["batch_index"],
            group["labels_cumulative"],
            marker="o",
            linewidth=1,
            label=f"seed={seed}",
        )
    axes[0].set_ylabel("Rolling PR-AUC")
    axes[1].set_ylabel("Cumulative labels")
    axes[1].set_xlabel("Batch")
    axes[0].set_title("Adaptive fraud detection quality and labeling cost")
    axes[0].legend()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"saved plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

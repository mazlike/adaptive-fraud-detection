"""Summarize factorial sweep results and run paired Wilcoxon checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


GROUP_COLUMNS = ["drift_type", "budget_fraction", "strategy", "variant"]
METRICS = ["pr_auc_overall", "recall_at_budget_overall", "labels_spent_total"]


def _paired_test(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
) -> dict[str, object]:
    keys = ["drift_type", "budget_fraction", "variant", "seed"]
    left_frame = frame[frame["strategy"] == left][keys + [metric]].rename(
        columns={metric: "left"}
    )
    right_frame = frame[frame["strategy"] == right][keys + [metric]].rename(
        columns={metric: "right"}
    )
    joined = left_frame.merge(right_frame, on=keys, how="inner").dropna()
    if len(joined) < 2:
        return {
            "left": left,
            "right": right,
            "metric": metric,
            "n_pairs": len(joined),
            "statistic": float("nan"),
            "p_value": float("nan"),
        }
    differences = joined["left"] - joined["right"]
    if (differences == 0).all():
        return {
            "left": left,
            "right": right,
            "metric": metric,
            "n_pairs": len(joined),
            "statistic": 0.0,
            "p_value": 1.0,
        }
    test = wilcoxon(joined["left"], joined["right"], zero_method="wilcox")
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "n_pairs": len(joined),
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze factorial fraud experiments")
    parser.add_argument("--results", required=True, help="Directory containing sweep_metrics.csv")
    args = parser.parse_args()
    results_dir = Path(args.results)
    source = results_dir / "sweep_metrics.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing sweep results: {source}")
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("sweep_metrics.csv contains no rows")

    summary = (
        frame.groupby(GROUP_COLUMNS, dropna=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(results_dir / "sweep_summary.csv", index=False)

    ranking = (
        frame.groupby(["drift_type", "budget_fraction", "strategy", "variant"], dropna=False)
        .agg(
            pr_auc_mean=("pr_auc_overall", "mean"),
            pr_auc_std=("pr_auc_overall", "std"),
            recall_mean=("recall_at_budget_overall", "mean"),
            labels_mean=("labels_spent_total", "mean"),
        )
        .reset_index()
        .sort_values(
            ["drift_type", "budget_fraction", "pr_auc_mean", "recall_mean"],
            ascending=[True, True, False, False],
        )
    )
    best = ranking.groupby(["drift_type", "budget_fraction"], as_index=False).first()
    best.to_csv(results_dir / "best_configurations.csv", index=False)

    tests: list[dict[str, object]] = []
    for drift_type in frame["drift_type"].unique():
        for budget in frame["budget_fraction"].unique():
            subset = frame[
                (frame["drift_type"] == drift_type)
                & (frame["budget_fraction"] == budget)
            ]
            for variant in frame["variant"].unique():
                variant_subset = subset[subset["variant"] == variant]
                for metric in METRICS[:2]:
                    for left, right in (("uncertainty", "random"), ("rare_aware", "uncertainty")):
                        result = _paired_test(variant_subset, left, right, metric)
                        result.update(
                            {
                                "drift_type": drift_type,
                                "budget_fraction": budget,
                                "variant": variant,
                            }
                        )
                        tests.append(result)
    pd.DataFrame(tests).to_csv(results_dir / "paired_wilcoxon.csv", index=False)
    print(f"saved analysis to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

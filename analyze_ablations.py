#!/usr/bin/env python3
"""
Ablation Study Analysis Script

Analyzes results from ablation experiments (none, q_only, p_only) and generates:
- Statistical comparisons with t-tests and effect sizes
- Confidence intervals for performance metrics
- Comparison visualizations

Usage:
    python analyze_ablations.py --experiment_dir ./experiments --output_dir ./ablation_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_experiment_results(experiment_dir: str) -> pd.DataFrame:
    """Load results from all ablation experiment directories."""
    results = []

    # Find all ablation experiment directories
    exp_path = Path(experiment_dir)
    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    ablation_dirs = list(exp_path.glob("ablation_*"))

    if not ablation_dirs:
        print(f"Error: No ablation experiments found in {experiment_dir}")
        print("Expected directories matching pattern: ablation_*/")
        sys.exit(1)

    print(f"Found {len(ablation_dirs)} experiment directories")

    for exp_dir in ablation_dirs:
        # Parse experiment name: ablation_<type>_seed<N>
        dir_name = exp_dir.name
        parts = dir_name.split("_")

        if len(parts) < 3:
            print(f"Warning: Skipping directory with unexpected name: {dir_name}")
            continue

        ablation_type = parts[1]  # none, q_only, or p_only
        seed_part = parts[2]  # seed42, seed43, etc.

        try:
            seed = int(seed_part.replace("seed", ""))
        except ValueError:
            print(f"Warning: Could not parse seed from {dir_name}")
            continue

        # Load metrics
        metrics_file = exp_dir / "eval_metrics.json"

        if not metrics_file.exists():
            print(f"Warning: No eval_metrics.json found in {exp_dir}")
            continue

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            results.append(
                {
                    "ablation": ablation_type,
                    "seed": seed,
                    "f1": metrics.get("eval_f1", None),
                    "exact_match": metrics.get("eval_exact_match", None),
                    "loss": metrics.get("eval_loss", None),
                    "experiment_dir": str(exp_dir),
                }
            )

            print(f"  ✓ Loaded: {dir_name}")

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {metrics_file}: {e}")
            continue

    if not results:
        print("Error: No valid experiment results loaded")
        sys.exit(1)

    df = pd.DataFrame(results)

    # Filter out any results with missing metrics
    original_len = len(df)
    df = df.dropna(subset=["f1", "exact_match"])

    if len(df) < original_len:
        print(
            f"Warning: Dropped {original_len - len(df)} experiments with missing metrics"
        )

    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, and confidence intervals for each ablation."""
    stats_list = []

    for ablation in df["ablation"].unique():
        ablation_df = df[df["ablation"] == ablation]

        for metric in ["f1", "exact_match"]:
            values = ablation_df[metric].values
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0

            # 95% confidence interval
            if n > 1:
                sem = stats.sem(values)
                ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
                ci_lower, ci_upper = ci
            else:
                ci_lower = ci_upper = mean

            stats_list.append(
                {
                    "ablation": ablation,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "values": values.tolist(),
                }
            )

    return pd.DataFrame(stats_list)


def compute_comparisons(df: pd.DataFrame) -> Dict:
    """Compute pairwise comparisons between ablations."""
    comparisons = {}

    # Get baseline (none) results
    baseline_df = df[df["ablation"] == "none"]

    if len(baseline_df) == 0:
        print("Warning: No baseline (none) results found for comparison")
        return comparisons

    for ablation in ["q_only", "p_only"]:
        ablation_df = df[df["ablation"] == ablation]

        if len(ablation_df) == 0:
            print(f"Warning: No {ablation} results found")
            continue

        comparisons[ablation] = {}

        for metric in ["f1", "exact_match"]:
            baseline_values = baseline_df[metric].values
            ablation_values = ablation_df[metric].values

            baseline_mean = np.mean(baseline_values)
            ablation_mean = np.mean(ablation_values)

            # Compute delta
            delta = ablation_mean - baseline_mean
            delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0

            # Paired t-test (if same number of seeds)
            if (
                len(baseline_values) == len(ablation_values)
                and len(baseline_values) > 1
            ):
                t_stat, p_value = stats.ttest_rel(ablation_values, baseline_values)
                test_type = "paired"
            elif len(baseline_values) > 1 and len(ablation_values) > 1:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(ablation_values, baseline_values)
                test_type = "independent"
            else:
                t_stat = None
                p_value = None
                test_type = "insufficient_data"

            # Cohen's d effect size
            if len(baseline_values) > 1 and len(ablation_values) > 1:
                pooled_std = np.sqrt(
                    (np.var(baseline_values, ddof=1) + np.var(ablation_values, ddof=1))
                    / 2
                )
                cohens_d = delta / pooled_std if pooled_std != 0 else 0
            else:
                cohens_d = None

            # Interpret effect size
            if cohens_d is not None:
                if abs(cohens_d) < 0.2:
                    effect_interpretation = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_interpretation = "small"
                elif abs(cohens_d) < 0.8:
                    effect_interpretation = "medium"
                else:
                    effect_interpretation = "large"
            else:
                effect_interpretation = "unknown"

            # Significance stars
            if p_value is not None:
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
            else:
                significance = "n/a"

            comparisons[ablation][metric] = {
                "baseline_mean": baseline_mean,
                "ablation_mean": ablation_mean,
                "delta": delta,
                "delta_pct": delta_pct,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significance": significance,
                "cohens_d": cohens_d,
                "effect_size": effect_interpretation,
                "test_type": test_type,
            }

    return comparisons


def print_summary(stats_df: pd.DataFrame, comparisons: Dict, output_dir: str):
    """Print formatted summary to console and save to file."""
    summary_lines = []

    def print_and_save(line=""):
        print(line)
        summary_lines.append(line)

    print_and_save("=" * 80)
    print_and_save("ABLATION STUDY RESULTS")
    print_and_save("=" * 80)
    print_and_save()

    # Performance by ablation
    print_and_save("Performance by Ablation Type:")
    print_and_save("-" * 80)

    for ablation in ["none", "q_only", "p_only"]:
        ablation_stats = stats_df[stats_df["ablation"] == ablation]

        if len(ablation_stats) == 0:
            continue

        print_and_save(f"\n{ablation.upper()}")

        for _, row in ablation_stats.iterrows():
            metric = row["metric"]
            mean = row["mean"]
            std = row["std"]
            ci_lower = row["ci_lower"]
            ci_upper = row["ci_upper"]
            n = row["n"]

            print_and_save(
                f"  {metric:15s}: {mean:.4f} ± {std:.4f}  "
                f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]  (n={n})"
            )

    # Comparisons with baseline
    if comparisons:
        print_and_save()
        print_and_save("=" * 80)
        print_and_save("COMPARISON WITH BASELINE (none)")
        print_and_save("=" * 80)

        for ablation in ["q_only", "p_only"]:
            if ablation not in comparisons:
                continue

            print_and_save(f"\n{ablation.upper()} vs NONE:")
            print_and_save("-" * 80)

            for metric in ["f1", "exact_match"]:
                if metric not in comparisons[ablation]:
                    continue

                comp = comparisons[ablation][metric]

                print_and_save(f"\n{metric.upper()}:")
                print_and_save(f"  Baseline:      {comp['baseline_mean']:.4f}")
                print_and_save(f"  {ablation:13s}: {comp['ablation_mean']:.4f}")
                print_and_save(
                    f"  Delta:         {comp['delta']:+.4f} ({comp['delta_pct']:+.2f}%)"
                )

                if comp["p_value"] is not None:
                    print_and_save(
                        f"  Significance:  p={comp['p_value']:.4f} {comp['significance']}"
                    )
                    print_and_save(f"  Test type:     {comp['test_type']}")

                if comp["cohens_d"] is not None:
                    print_and_save(
                        f"  Effect size:   d={comp['cohens_d']:.3f} ({comp['effect_size']})"
                    )

    print_and_save()
    print_and_save("=" * 80)
    print_and_save("LEGEND")
    print_and_save("=" * 80)
    print_and_save("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print_and_save(
        "Effect size (Cohen's d): <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large"
    )
    print_and_save("=" * 80)

    # Save to file
    summary_file = Path(output_dir) / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSummary saved to: {summary_file}")


def plot_ablation_comparison(stats_df: pd.DataFrame, output_dir: str):
    """Create grouped bar chart comparing ablations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ["f1", "exact_match"]
    metric_labels = ["F1 Score", "Exact Match"]
    ablations = ["none", "q_only", "p_only"]
    ablation_labels = ["Full Model\n(Baseline)", "Question\nOnly", "Passage\nOnly"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    x = np.arange(len(ablations))
    width = 0.6

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        means = []
        stds = []
        cis_lower = []
        cis_upper = []

        for ablation in ablations:
            ablation_stats = stats_df[
                (stats_df["ablation"] == ablation) & (stats_df["metric"] == metric)
            ]

            if len(ablation_stats) > 0:
                row = ablation_stats.iloc[0]
                means.append(row["mean"])
                stds.append(row["std"])
                cis_lower.append(row["mean"] - row["ci_lower"])
                cis_upper.append(row["ci_upper"] - row["mean"])
            else:
                means.append(0)
                stds.append(0)
                cis_lower.append(0)
                cis_upper.append(0)

        # Create bars
        bars = ax.bar(
            x,
            means,
            width,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Add error bars (using confidence intervals)
        ax.errorbar(
            x,
            means,
            yerr=[cis_lower, cis_upper],
            fmt="none",
            ecolor="black",
            elinewidth=2,
            capsize=5,
            capthick=2,
        )

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + cis_upper[i] + 0.01,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Ablation Type", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{metric_label} by Ablation",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_labels, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Set y-axis to start from reasonable minimum
        y_min = min(means) - 0.1
        y_max = max(means) + max(cis_upper) + 0.1
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    output_file = Path(output_dir) / "ablation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved comparison plot: {output_file}")
    plt.close()


def plot_ablation_deltas(comparisons: Dict, output_dir: str):
    """Create bar chart showing performance deltas from baseline."""
    if not comparisons:
        print("  ⚠ Skipping delta plot (no comparisons available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ["f1", "exact_match"]
    metric_labels = ["F1 Score", "Exact Match"]
    ablations = ["q_only", "p_only"]
    ablation_labels = ["Question Only", "Passage Only"]
    colors = ["#A23B72", "#F18F01"]

    x = np.arange(len(ablations))
    width = 0.6

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        deltas = []
        significances = []

        for ablation in ablations:
            if ablation in comparisons and metric in comparisons[ablation]:
                comp = comparisons[ablation][metric]
                deltas.append(comp["delta_pct"])
                significances.append(comp["significance"])
            else:
                deltas.append(0)
                significances.append("n/a")

        # Create bars
        bars = ax.bar(
            x,
            deltas,
            width,
            color=[colors[i] if d < 0 else "#90BE6D" for i, d in enumerate(deltas)],
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Add value labels and significance stars
        for i, (bar, delta, sig) in enumerate(zip(bars, deltas, significances)):
            height = bar.get_height()

            # Position label above or below bar depending on sign
            if height >= 0:
                va = "bottom"
                y_pos = height + 0.5
            else:
                va = "top"
                y_pos = height - 0.5

            label = f"{delta:+.1f}%"
            if sig != "n/a" and sig != "ns":
                label += f"\n{sig}"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                label,
                ha="center",
                va=va,
                fontsize=11,
                fontweight="bold",
            )

        # Add zero line
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7)

        ax.set_ylabel("Change from Baseline (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Ablation Type", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{metric_label} Change vs Baseline",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_labels, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()

    output_file = Path(output_dir) / "ablation_deltas.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved delta plot: {output_file}")
    plt.close()


def plot_seed_scatter(df: pd.DataFrame, output_dir: str):
    """Create scatter plot showing individual seed results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ["f1", "exact_match"]
    metric_labels = ["F1 Score", "Exact Match"]
    ablations = ["none", "q_only", "p_only"]
    ablation_labels = ["Full Model", "Question Only", "Passage Only"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    markers = ["o", "s", "^"]

    x_positions = {"none": 1, "q_only": 2, "p_only": 3}

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        # Plot individual points
        for ablation, color, marker, label in zip(
            ablations, colors, markers, ablation_labels
        ):
            ablation_df = df[df["ablation"] == ablation]

            if len(ablation_df) > 0:
                x = [x_positions[ablation]] * len(ablation_df)
                # Add small jitter for visibility
                x = np.array(x) + np.random.normal(0, 0.05, len(x))
                y = ablation_df[metric].values

                ax.scatter(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    s=100,
                    alpha=0.6,
                    edgecolors="black",
                    linewidth=1.5,
                    label=label,
                )

                # Add mean line
                mean_val = np.mean(y)
                ax.hlines(
                    mean_val,
                    x_positions[ablation] - 0.3,
                    x_positions[ablation] + 0.3,
                    colors=color,
                    linewidth=3,
                    alpha=0.8,
                )

        ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Ablation Type", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{metric_label} by Seed",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(ablation_labels, fontsize=10)
        ax.set_xlim(0.5, 3.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()

    output_file = Path(output_dir) / "seed_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved scatter plot: {output_file}")
    plt.close()


def export_results(
    df: pd.DataFrame, stats_df: pd.DataFrame, comparisons: Dict, output_dir: str
):
    """Export results to CSV and JSON."""
    output_path = Path(output_dir)

    # Export summary statistics to CSV
    summary_csv = output_path / "summary.csv"

    # Reshape stats for CSV
    summary_rows = []
    for ablation in stats_df["ablation"].unique():
        row = {"ablation": ablation}
        ablation_stats = stats_df[stats_df["ablation"] == ablation]

        for _, stat_row in ablation_stats.iterrows():
            metric = stat_row["metric"]
            row[f"{metric}_mean"] = stat_row["mean"]
            row[f"{metric}_std"] = stat_row["std"]
            row[f"{metric}_ci_lower"] = stat_row["ci_lower"]
            row[f"{metric}_ci_upper"] = stat_row["ci_upper"]
            row[f"{metric}_n"] = stat_row["n"]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"  ✓ Saved summary CSV: {summary_csv}")

    # Export detailed results to JSON
    detailed_json = output_path / "detailed_results.json"

    detailed_results = {
        "raw_data": df.to_dict(orient="records"),
        "statistics": stats_df.to_dict(orient="records"),
        "comparisons": comparisons,
    }

    with open(detailed_json, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"  ✓ Saved detailed JSON: {detailed_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablation study results with statistical tests"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="./experiments",
        help="Directory containing ablation experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_results",
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 80)
    print()

    # Load results
    print("Loading experiment results...")
    df = load_experiment_results(args.experiment_dir)
    print(f"  ✓ Loaded {len(df)} experiment results")
    print()

    # Compute statistics
    print("Computing statistics...")
    stats_df = compute_statistics(df)
    print(f"  ✓ Computed statistics for {len(stats_df)} ablation-metric combinations")
    print()

    # Compute comparisons
    print("Computing pairwise comparisons...")
    comparisons = compute_comparisons(df)
    print(f"  ✓ Completed {len(comparisons)} comparisons")
    print()

    # Print summary
    print_summary(stats_df, comparisons, args.output_dir)
    print()

    # Generate visualizations
    print("Generating visualizations...")
    plot_ablation_comparison(stats_df, args.output_dir)
    plot_ablation_deltas(comparisons, args.output_dir)
    plot_seed_scatter(df, args.output_dir)
    print()

    # Export results
    print("Exporting results...")
    export_results(df, stats_df, comparisons, args.output_dir)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

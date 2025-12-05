#!/usr/bin/env python3
"""
Analyze and compare results from W&B hyperparameter sweeps.

This script provides utilities to:
- Download and aggregate results from multiple sweeps
- Compare performance across different strategies
- Generate comparison plots and tables
- Export results for reporting

Usage:
    # Compare all sweeps in project
    python analyze_sweep_results.py --project qa-cartography-experiments

    # Compare specific sweeps
    python analyze_sweep_results.py --sweep_ids abc123 def456 ghi789

    # Compare with baseline
    python analyze_sweep_results.py --compare_with_baseline

    # Export results to CSV
    python analyze_sweep_results.py --export results_comparison.csv
"""

import argparse
import sys

from typing import Dict, List

import pandas as pd
import wandb


def get_sweep_runs(sweep_id: str, project: str) -> List[wandb.apis.public.Run]:
    """Get all runs from a sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    return list(sweep.runs)


def extract_run_metrics(run: wandb.apis.public.Run) -> Dict:
    """Extract key metrics and configuration from a run."""
    config = run.config
    summary = run.summary

    return {
        # Run metadata
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "tags": run.tags,
        # Training config
        "model": config.get("model", "N/A"),
        "dataset": config.get("dataset", "N/A"),
        "num_epochs": config.get("num_train_epochs", "N/A"),
        "batch_size": config.get("per_device_train_batch_size", "N/A"),
        "learning_rate": config.get("learning_rate", "N/A"),
        "seed": config.get("seed", "N/A"),
        # Filtering strategies
        "filter_ambiguous": config.get("filter_ambiguous", False),
        "ambiguous_top_fraction": config.get("ambiguous_top_fraction", "N/A"),
        "variability_margin": config.get("variability_margin", "N/A"),
        "filter_clusters": config.get("filter_clusters", False),
        "exclude_noise_cluster": config.get("exclude_noise_cluster", False),
        "min_cluster_probability": config.get("min_cluster_probability", "N/A"),
        "filter_rule_based": config.get("filter_rule_based", False),
        "rule_name": config.get("rule_name", "N/A"),
        "rule_sim_threshold": config.get("rule_sim_threshold", "N/A"),
        "filter_validation": config.get("filter_validation", False),
        # Training modifications
        "use_label_smoothing": config.get("use_label_smoothing", False),
        "smoothing_factor": config.get("smoothing_factor", "N/A"),
        "use_soft_weighting": config.get("use_soft_weighting", False),
        "weight_clip_min": config.get("weight_clip_min", "N/A"),
        "weight_clip_max": config.get("weight_clip_max", "N/A"),
        # Performance metrics
        "eval_f1": summary.get("eval/f1", None),
        "eval_exact_match": summary.get("eval/exact_match", None),
        "eval_loss": summary.get("eval/loss", None),
        "train_loss": summary.get("train/loss", None),
        # Dataset sizes (if logged)
        "train_original_size": summary.get("train/original_size", "N/A"),
        "train_filtered_size": summary.get("train/filtered_size", "N/A"),
        "train_removal_percentage": summary.get("train/removal_percentage", "N/A"),
    }


def analyze_project(project: str, baseline_only: bool = False) -> pd.DataFrame:
    """Analyze all runs in a project."""
    api = wandb.Api()

    # Get all runs
    runs = api.runs(project)

    results = []
    for run in runs:
        if baseline_only and "baseline" not in run.tags:
            continue

        if run.state == "finished":
            metrics = extract_run_metrics(run)
            results.append(metrics)

    df = pd.DataFrame(results)
    return df


def analyze_sweeps(sweep_ids: List[str], project: str) -> pd.DataFrame:
    """Analyze specific sweeps."""
    all_results = []

    for sweep_id in sweep_ids:
        print(f"Analyzing sweep: {sweep_id}")
        runs = get_sweep_runs(sweep_id, project)

        for run in runs:
            if run.state == "finished":
                metrics = extract_run_metrics(run)
                metrics["sweep_id"] = sweep_id
                all_results.append(metrics)

    df = pd.DataFrame(all_results)
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for different strategies."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall statistics
    print(f"\nTotal runs analyzed: {len(df)}")
    print(f"Finished runs: {len(df[df['state'] == 'finished'])}")

    # Group by strategy type
    strategies = {
        "Baseline": (~df["filter_ambiguous"])
        & (~df["filter_clusters"])
        & (~df["filter_rule_based"])
        & (~df["use_label_smoothing"])
        & (~df["use_soft_weighting"]),
        "Ambiguous Filtering": df["filter_ambiguous"],
        "Cluster Filtering": df["filter_clusters"],
        "Rule-Based Filtering": df["filter_rule_based"],
        "Label Smoothing": df["use_label_smoothing"],
        "Soft Weighting": df["use_soft_weighting"],
    }

    print("\n" + "-" * 80)
    print("Performance by Strategy")
    print("-" * 80)
    print(
        f"{'Strategy':<30} {'Count':<10} {'Mean F1':<12} {'Std F1':<12} {'Best F1':<12}"
    )
    print("-" * 80)

    for strategy_name, mask in strategies.items():
        strategy_df = df[mask]
        if len(strategy_df) > 0 and strategy_df["eval_f1"].notna().any():
            count = len(strategy_df)
            mean_f1 = strategy_df["eval_f1"].mean()
            std_f1 = strategy_df["eval_f1"].std()
            best_f1 = strategy_df["eval_f1"].max()

            print(
                f"{strategy_name:<30} {count:<10} {mean_f1:<12.4f} {std_f1:<12.4f} {best_f1:<12.4f}"
            )

    print("-" * 80)


def compare_with_baseline(df: pd.DataFrame):
    """Compare all strategies against baseline."""
    baseline_mask = (
        (~df["filter_ambiguous"])
        & (~df["filter_clusters"])
        & (~df["filter_rule_based"])
        & (~df["use_label_smoothing"])
        & (~df["use_soft_weighting"])
    )

    baseline_df = df[baseline_mask]

    if len(baseline_df) == 0:
        print("\nNo baseline runs found!")
        return

    baseline_f1_mean = baseline_df["eval_f1"].mean()
    baseline_f1_std = baseline_df["eval_f1"].std()

    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    print("\nBaseline Performance:")
    print(f"  Mean F1: {baseline_f1_mean:.4f} ± {baseline_f1_std:.4f}")
    print(f"  Best F1: {baseline_df['eval_f1'].max():.4f}")
    print(f"  Runs: {len(baseline_df)}")

    # Find best runs for each strategy
    print("\n" + "-" * 80)
    print("Best Runs by Strategy (vs Baseline)")
    print("-" * 80)

    # Group by combinations of strategies
    grouped = df.groupby(
        [
            "filter_ambiguous",
            "filter_clusters",
            "filter_rule_based",
            "use_label_smoothing",
            "use_soft_weighting",
        ]
    )

    improvements = []

    for name, group in grouped:
        if len(group) > 0 and group["eval_f1"].notna().any():
            best_run = group.loc[group["eval_f1"].idxmax()]
            best_f1 = best_run["eval_f1"]
            improvement = ((best_f1 - baseline_f1_mean) / baseline_f1_mean) * 100

            strategy_desc = []
            if name[0]:
                strategy_desc.append("AmbigFilt")
            if name[1]:
                strategy_desc.append("ClustFilt")
            if name[2]:
                strategy_desc.append("RuleFilt")
            if name[3]:
                strategy_desc.append("Smooth")
            if name[4]:
                strategy_desc.append("Weight")

            strategy_str = "+".join(strategy_desc) if strategy_desc else "Baseline"

            improvements.append(
                {
                    "Strategy": strategy_str,
                    "Best F1": best_f1,
                    "Improvement": improvement,
                    "Run": best_run["run_name"],
                }
            )

    # Sort by improvement
    improvements_df = pd.DataFrame(improvements).sort_values(
        "Improvement", ascending=False
    )

    for _, row in improvements_df.iterrows():
        print(
            f"{row['Strategy']:<40} F1: {row['Best F1']:.4f} ({row['Improvement']:+.2f}%)"
        )

    print("-" * 80)


def export_results(df: pd.DataFrame, output_path: str):
    """Export results to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze W&B hyperparameter sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--project",
        type=str,
        default="qa-cartography-experiments",
        help="W&B project name",
    )

    parser.add_argument("--sweep_ids", nargs="+", help="Specific sweep IDs to analyze")

    parser.add_argument(
        "--compare_with_baseline",
        action="store_true",
        help="Compare all strategies with baseline",
    )

    parser.add_argument(
        "--export", type=str, metavar="OUTPUT_PATH", help="Export results to CSV file"
    )

    parser.add_argument(
        "--baseline_only", action="store_true", help="Only analyze baseline runs"
    )

    args = parser.parse_args()

    # Fetch results
    if args.sweep_ids:
        df = analyze_sweeps(args.sweep_ids, args.project)
    else:
        df = analyze_project(args.project, args.baseline_only)

    if len(df) == 0:
        print("No finished runs found!")
        sys.exit(1)

    # Print summary
    print_summary_statistics(df)

    # Compare with baseline if requested
    if args.compare_with_baseline:
        compare_with_baseline(df)

    # Export if requested
    if args.export:
        export_results(df, args.export)

    # Show top 10 runs
    print("\n" + "=" * 80)
    print("TOP 10 RUNS BY F1 SCORE")
    print("=" * 80)

    top_runs = df.nlargest(10, "eval_f1")[
        [
            "run_name",
            "eval_f1",
            "filter_ambiguous",
            "filter_clusters",
            "filter_rule_based",
            "use_label_smoothing",
            "use_soft_weighting",
        ]
    ]

    print(top_runs.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

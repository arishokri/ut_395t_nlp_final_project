"""
Experiment Analysis Tools

This module provides utilities for analyzing and visualizing experiment results.
"""

import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_experiment_results(experiments_dir: str = "./experiments") -> pd.DataFrame:
    """
    Load all experiment results into a DataFrame.

    Args:
        experiments_dir: Directory containing experiment outputs

    Returns:
        DataFrame with one row per experiment
    """
    results = []

    # Look for all results.json files in experiment directories
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)

        if not os.path.isdir(exp_path):
            continue

        results_file = os.path.join(exp_path, "results.json")
        config_file = os.path.join(exp_path, "config.json")

        if not os.path.exists(results_file):
            continue

        try:
            # Load results
            with open(results_file, "r") as f:
                result = json.load(f)

            # Load config for additional context
            config = {}
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)

            # Flatten nested structure for DataFrame
            row = {
                "experiment_name": result.get("experiment_name"),
                "config_hash": result.get("config_hash"),
                "success": result.get("success"),
                "duration_seconds": result.get("duration_seconds"),
                "timestamp": result.get("timestamp"),
                "error": result.get("error"),
            }

            # Add metrics
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                row[key] = value

            # Add key config parameters
            training_config = config.get("training", {})
            row["seed"] = training_config.get("seed")
            row["num_train_epochs"] = training_config.get("num_train_epochs")
            row["learning_rate"] = training_config.get("learning_rate")
            row["max_train_samples"] = training_config.get("max_train_samples")
            row["max_eval_samples"] = training_config.get("max_eval_samples")

            # Add filtering/strategy flags
            row["cartography_filter"] = training_config.get(
                "cartography_filter", {}
            ).get("enabled", False)
            row["cluster_filter"] = training_config.get("cluster_filter", {}).get(
                "enabled", False
            )
            row["label_smoothing"] = training_config.get("label_smoothing", {}).get(
                "enabled", False
            )
            row["soft_weighting"] = training_config.get("soft_weighting", {}).get(
                "enabled", False
            )

            results.append(row)

        except Exception as e:
            print(f"[WARNING] Failed to load experiment {exp_dir}: {e}")

    if not results:
        print(f"[WARNING] No experiment results found in {experiments_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    return df


def print_results_summary(df: pd.DataFrame):
    """
    Print a summary of experiment results.

    Args:
        df: DataFrame with experiment results
    """
    if df.empty:
        print("No results to summarize")
        return

    print("=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

    # Overall statistics
    total = len(df)
    successful = df["success"].sum()
    failed = total - successful

    print(f"\nTotal experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful == 0:
        print("\n[WARNING] No successful experiments to analyze")
        return

    # Filter to successful experiments
    df_success = df[df["success"]].copy()

    # Metrics summary
    print("\n" + "-" * 100)
    print("PERFORMANCE METRICS (successful experiments only)")
    print("-" * 100)

    metric_cols = [col for col in df_success.columns if col.startswith("eval_")]
    if metric_cols:
        print(df_success[metric_cols].describe().round(4).to_string())
    else:
        print("No evaluation metrics found")

    # Best experiments
    print("\n" + "-" * 100)
    print("TOP 5 EXPERIMENTS BY F1 SCORE")
    print("-" * 100)

    if "eval_f1" in df_success.columns:
        top_experiments = df_success.nlargest(5, "eval_f1")

        display_cols = [
            "experiment_name",
            "eval_f1",
            "eval_exact_match",
            "duration_seconds",
        ]
        display_cols = [c for c in display_cols if c in top_experiments.columns]

        print(top_experiments[display_cols].to_string(index=False))
    else:
        print("F1 metric not available")

    # Strategy comparison
    print("\n" + "-" * 100)
    print("PERFORMANCE BY STRATEGY")
    print("-" * 100)

    strategy_cols = [
        "cartography_filter",
        "cluster_filter",
        "label_smoothing",
        "soft_weighting",
    ]

    for col in strategy_cols:
        if col not in df_success.columns:
            continue

        enabled = df_success[df_success[col]]
        disabled = df_success[~df_success[col]]

        if len(enabled) > 0 and len(disabled) > 0 and "eval_f1" in df_success.columns:
            print(f"\n{col.replace('_', ' ').title()}:")
            print(
                f"  Enabled:  {len(enabled):3d} experiments | "
                f"Mean F1: {enabled['eval_f1'].mean():.4f} | "
                f"Std: {enabled['eval_f1'].std():.4f}"
            )
            print(
                f"  Disabled: {len(disabled):3d} experiments | "
                f"Mean F1: {disabled['eval_f1'].mean():.4f} | "
                f"Std: {disabled['eval_f1'].std():.4f}"
            )

    print("\n" + "=" * 100)


def plot_results(
    df: pd.DataFrame,
    output_dir: str = "./experiments",
    show_plots: bool = False,
):
    """
    Generate visualizations of experiment results.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
    """
    if df.empty or df["success"].sum() == 0:
        print("No successful experiments to plot")
        return

    # Filter to successful experiments
    df_success = df[df["success"]].copy()

    # Create figures
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment Results Analysis", fontsize=16, fontweight="bold")

    # Plot 1: F1 scores over time
    ax = axes[0, 0]
    if "eval_f1" in df_success.columns and "timestamp" in df_success.columns:
        df_plot = df_success.sort_values("timestamp").reset_index(drop=True)
        ax.plot(
            range(len(df_plot)),
            df_plot["eval_f1"],
            marker="o",
            linestyle="-",
            alpha=0.7,
        )
        ax.set_xlabel("Experiment Number")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score Over Experiments")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "F1 data not available", ha="center", va="center")

    # Plot 2: F1 vs Exact Match
    ax = axes[0, 1]
    if "eval_f1" in df_success.columns and "eval_exact_match" in df_success.columns:
        ax.scatter(
            df_success["eval_exact_match"],
            df_success["eval_f1"],
            alpha=0.6,
            s=100,
        )
        ax.set_xlabel("Exact Match")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 vs Exact Match")
        ax.grid(True, alpha=0.3)

        # Add diagonal line
        min_val = min(df_success["eval_exact_match"].min(), df_success["eval_f1"].min())
        max_val = max(df_success["eval_exact_match"].max(), df_success["eval_f1"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Metrics not available", ha="center", va="center")

    # Plot 3: Strategy comparison
    ax = axes[1, 0]
    strategy_cols = [
        "cartography_filter",
        "cluster_filter",
        "label_smoothing",
        "soft_weighting",
    ]

    if "eval_f1" in df_success.columns:
        strategy_means = []
        strategy_labels = []

        for col in strategy_cols:
            if col in df_success.columns:
                enabled = df_success[df_success[col]]
                if len(enabled) > 0:
                    strategy_means.append(enabled["eval_f1"].mean())
                    strategy_labels.append(col.replace("_", " ").title())

        if strategy_means:
            bars = ax.barh(strategy_labels, strategy_means, alpha=0.7)
            ax.set_xlabel("Mean F1 Score")
            ax.set_title("Strategy Performance Comparison")
            ax.grid(True, alpha=0.3, axis="x")

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, strategy_means)):
                ax.text(val, i, f" {val:.4f}", va="center")
        else:
            ax.text(0.5, 0.5, "No strategy data", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "F1 data not available", ha="center", va="center")

    # Plot 4: Duration vs Performance
    ax = axes[1, 1]
    if "eval_f1" in df_success.columns and "duration_seconds" in df_success.columns:
        # Convert duration to minutes
        durations_min = df_success["duration_seconds"] / 60

        ax.scatter(
            durations_min,
            df_success["eval_f1"],
            alpha=0.6,
            s=100,
        )
        ax.set_xlabel("Training Duration (minutes)")
        ax.set_ylabel("F1 Score")
        ax.set_title("Training Efficiency")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Duration data not available", ha="center", va="center")

    plt.tight_layout()

    # Save figure
    plot_file = os.path.join(output_dir, "experiment_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\n[SAVED] Plots saved to: {plot_file}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def export_results_csv(
    df: pd.DataFrame, output_file: str = "./experiments/results.csv"
):
    """
    Export experiment results to CSV.

    Args:
        df: DataFrame with experiment results
        output_file: Path to output CSV file
    """
    if df.empty:
        print("No results to export")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"[SAVED] Results exported to: {output_file}")


def compare_experiments(
    df: pd.DataFrame,
    experiment_names: List[str],
    metrics: Optional[List[str]] = None,
):
    """
    Compare specific experiments side-by-side.

    Args:
        df: DataFrame with experiment results
        experiment_names: List of experiment names to compare
        metrics: List of metrics to compare (default: eval metrics)
    """
    if df.empty:
        print("No results to compare")
        return

    # Filter to specified experiments
    df_compare = df[df["experiment_name"].isin(experiment_names)].copy()

    if df_compare.empty:
        print(f"No experiments found matching: {experiment_names}")
        return

    # Select metrics to display
    if metrics is None:
        metrics = [col for col in df_compare.columns if col.startswith("eval_")]

    # Add some standard columns
    display_cols = ["experiment_name", "success", "duration_seconds"] + metrics

    # Filter to available columns
    display_cols = [c for c in display_cols if c in df_compare.columns]

    print("=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)

    print(df_compare[display_cols].to_string(index=False))

    print("=" * 100)


def analyze_all_experiments(
    experiments_dir: str = "./experiments", show_plots: bool = False
):
    """
    Load and analyze all experiments in directory.

    Args:
        experiments_dir: Directory containing experiment outputs
        show_plots: Whether to display plots interactively
    """
    print(f"\nLoading experiments from: {experiments_dir}")

    df = load_experiment_results(experiments_dir)

    if df.empty:
        print("No experiments found")
        return

    print(f"Loaded {len(df)} experiments\n")

    # Print summary
    print_results_summary(df)

    # Generate plots
    plot_results(df, output_dir=experiments_dir, show_plots=show_plots)

    # Export CSV
    export_results_csv(df, output_file=os.path.join(experiments_dir, "results.csv"))

    return df

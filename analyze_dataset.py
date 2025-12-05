"""
Unified Dataset Analysis Script

Analyzes both dataset cartography and clustering results,
and provides integrated visualizations showing the relationship
between cartography categories and semantic clusters.

Usage:
    # Cartography only
    python analyze_dataset.py --cartography_dir ./cartography_output

    # Clustering only
    python analyze_dataset.py --cluster_dir ./cluster_output

    # Both (integrated analysis)
    python analyze_dataset.py --cartography_dir ./cartography_output --cluster_dir ./cluster_output
"""

import argparse
import json
import os

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_cartography import (
    analyze_cartography_by_question_type,
    categorize_examples,
    get_examples_by_category,
    load_cartography_metrics,
)
from cluster_analysis import load_cluster_assignments
from rule_based_errors import apply_rules_to_dataset
from helpers import generate_hash_ids


def analyze_cartography_only(
    cartography_dir: str,
    dataset_name: str,
    split: str,
    n_examples: int,
    output_dir: str,
):
    """Perform cartography-only analysis (original functionality)."""

    print("=" * 70)
    print("DATASET CARTOGRAPHY ANALYSIS")
    print("=" * 70)

    # Load cartography metrics
    print(f"\n1. Loading cartography metrics from {cartography_dir}...")
    cartography_df = load_cartography_metrics(cartography_dir)
    print(f"   Loaded metrics for {len(cartography_df)} examples")

    # Categorize examples
    print("\n2. Categorizing examples...")
    cartography_df = categorize_examples(cartography_df)

    category_counts = cartography_df["category"].value_counts()
    print("\n   Category distribution:")
    for cat, count in category_counts.items():
        print(f"   - {cat:15s}: {count:6d} ({100 * count / len(cartography_df):5.1f}%)")

    # Load original dataset
    print(f"\n3. Loading original dataset: {dataset_name}...")
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=dataset_name)
        dataset_split = dataset["train"]
    else:
        dataset = datasets.load_dataset(dataset_name)
        dataset_split = dataset[split]

    dataset_df = pd.DataFrame(dataset_split)

    # Generate IDs if missing
    if "id" not in dataset_df.columns:
        print("   Generating IDs for dataset examples...")
        dataset_with_ids = dataset_split.map(generate_hash_ids)
        dataset_df = pd.DataFrame(dataset_with_ids)

    print(f"   Loaded {len(dataset_df)} examples from {split} split")

    # Analyze by question type
    print("\n4. Analyzing metrics by question type...")
    question_type_analysis = analyze_cartography_by_question_type(
        cartography_df, dataset_df
    )
    print("\n" + str(question_type_analysis))

    # Save question type analysis
    qtype_file = os.path.join(output_dir, "question_type_analysis.csv")
    question_type_analysis.to_csv(qtype_file)
    print(f"\n   Saved to {qtype_file}")

    # Get example samples from each category
    print(f"\n5. Extracting example samples ({n_examples} per category)...")

    samples = {}
    for category in ["easy", "hard", "ambiguous"]:
        example_ids = get_examples_by_category(cartography_df, category, n=n_examples)

        # Get full examples from dataset
        category_examples = []
        for ex_id in example_ids:
            matching = dataset_df[dataset_df["id"] == ex_id]
            if len(matching) > 0:
                ex = matching.iloc[0].to_dict()
                # Add cartography metrics
                metrics = cartography_df.loc[ex_id].to_dict()
                ex["cartography_metrics"] = metrics
                category_examples.append(ex)

        samples[category] = category_examples
        print(f"   - {category:12s}: {len(category_examples)} examples")

    # Save samples
    samples_file = os.path.join(output_dir, "category_samples.json")
    with open(samples_file, "w") as f:
        json.dump(samples, f, indent=2, default=str)
    print(f"\n   Saved samples to {samples_file}")

    # Create visualizations
    print("\n6. Creating visualizations...")
    create_cartography_plots(cartography_df, output_dir)

    print("\n" + "=" * 70)
    print("CARTOGRAPHY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"- Question type analysis: {qtype_file}")
    print(f"- Category samples: {samples_file}")
    print(f"- Visualizations: {output_dir}/*.png")
    print("=" * 70 + "\n")


def analyze_clustering_only(
    cluster_dir: str,
    output_dir: str,
):
    """Perform clustering-only analysis."""

    print("=" * 70)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("=" * 70)

    # Load cluster results
    print(f"\n1. Loading cluster results from {cluster_dir}...")

    assignments_file = os.path.join(cluster_dir, "cluster_assignments.csv")
    stats_file = os.path.join(cluster_dir, "cluster_statistics.csv")
    metadata_file = os.path.join(cluster_dir, "cluster_metadata.json")

    if not os.path.exists(assignments_file):
        raise FileNotFoundError(f"Cluster assignments not found: {assignments_file}")

    cluster_df = pd.read_csv(assignments_file)
    cluster_stats = pd.read_csv(stats_file)

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    print(f"   Loaded {len(cluster_df)} examples")
    print(f"   Number of clusters: {metadata['n_clusters']}")
    print(f"   Clustering method: {metadata['clustering_method']}")

    print("\n2. Cluster Statistics:")
    print(cluster_stats.to_string(index=False))

    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults already saved in: {cluster_dir}")
    print("=" * 70 + "\n")


def analyze_integrated(
    cartography_dir: str,
    cluster_dir: str,
    dataset_name: str,
    split: str,
    output_dir: str,
):
    """Perform integrated analysis of cartography and clustering."""

    print("=" * 70)
    print("INTEGRATED CARTOGRAPHY + CLUSTERING ANALYSIS")
    print("=" * 70)

    # Load cartography metrics
    print(f"\n1. Loading cartography metrics from {cartography_dir}...")
    cartography_df = load_cartography_metrics(cartography_dir)
    cartography_df = categorize_examples(cartography_df)
    print(f"   Loaded metrics for {len(cartography_df)} examples")

    # Load cluster results
    print(f"\n2. Loading cluster results from {cluster_dir}...")
    assignments_file = os.path.join(cluster_dir, "cluster_assignments.csv")
    cluster_df = pd.read_csv(assignments_file, index_col="id")
    print(f"   Loaded {len(cluster_df)} cluster assignments")

    # Merge cartography and cluster data
    print("\n3. Merging cartography and cluster data...")
    merged_df = cartography_df.join(cluster_df, how="inner")
    print(f"   Merged {len(merged_df)} examples")

    if len(merged_df) == 0:
        print("\n[WARNING] No overlapping examples found!")
        print("Make sure cartography and clustering were run on the same dataset.")
        return

    # Load original dataset for context
    print(f"\n4. Loading original dataset: {dataset_name}...")
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=dataset_name)
        dataset_split = dataset["train"]
    else:
        dataset = datasets.load_dataset(dataset_name)
        dataset_split = dataset[split]

    dataset_df = pd.DataFrame(dataset_split)
    if "id" not in dataset_df.columns:
        dataset_with_ids = dataset_split.map(generate_hash_ids)
        dataset_df = pd.DataFrame(dataset_with_ids)

    # Analyze overlap between categories and clusters
    print("\n5. Analyzing cartography-cluster overlap...")
    overlap_analysis = analyze_category_cluster_overlap(merged_df)
    print("\nCartography Category vs Cluster Distribution:")
    print(overlap_analysis.to_string())

    # Save overlap analysis
    overlap_file = os.path.join(output_dir, "category_cluster_overlap.csv")
    overlap_analysis.to_csv(overlap_file)
    print(f"\nSaved overlap analysis to {overlap_file}")

    # Compute cluster-level cartography statistics
    print("\n6. Computing cluster-level cartography statistics...")
    cluster_cart_stats = (
        merged_df.groupby("cluster")
        .agg(
            {
                "confidence": ["mean", "std", "min", "max"],
                "variability": ["mean", "std", "min", "max"],
                "correctness": ["mean", "std", "min", "max"],
            }
        )
        .round(3)
    )

    print("\nCartography Statistics by Cluster:")
    print(cluster_cart_stats.to_string())

    stats_file = os.path.join(output_dir, "cluster_cartography_stats.csv")
    cluster_cart_stats.to_csv(stats_file)
    print(f"\nSaved to {stats_file}")

    # Find interesting patterns
    print("\n7. Identifying interesting patterns...")
    patterns = identify_patterns(merged_df, dataset_df)

    patterns_file = os.path.join(output_dir, "interesting_patterns.json")
    with open(patterns_file, "w") as f:
        json.dump(patterns, f, indent=2, default=str)
    print(f"   Saved patterns to {patterns_file}")

    # Create integrated visualizations
    print("\n8. Creating integrated visualizations...")
    create_integrated_visualizations(merged_df, output_dir)

    print("\n" + "=" * 70)
    print("INTEGRATED ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"- Category-cluster overlap: {overlap_file}")
    print(f"- Cluster cartography stats: {stats_file}")
    print(f"- Interesting patterns: {patterns_file}")
    print(f"- Visualizations: {output_dir}/*.png")
    print("=" * 70 + "\n")


def analyze_category_cluster_overlap(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how cartography categories overlap with clusters."""

    # Create crosstab
    overlap = pd.crosstab(
        merged_df["category"], merged_df["cluster"], margins=True, margins_name="Total"
    )

    return overlap


def identify_patterns(merged_df: pd.DataFrame, dataset_df: pd.DataFrame) -> dict:
    """Identify interesting patterns in the merged data."""

    patterns = {}

    # Pattern 1: Clusters with high proportion of ambiguous examples
    cluster_ambiguous = merged_df.groupby("cluster")["category"].apply(
        lambda x: (x == "ambiguous").mean()
    )
    high_ambiguous_clusters = cluster_ambiguous[cluster_ambiguous > 0.5].sort_values(
        ascending=False
    )

    patterns["high_ambiguous_clusters"] = {
        "description": "Clusters with >50% ambiguous examples",
        "clusters": high_ambiguous_clusters.to_dict(),
    }

    # Pattern 2: Clusters with high proportion of hard examples
    cluster_hard = merged_df.groupby("cluster")["category"].apply(
        lambda x: (x == "hard").mean()
    )
    high_hard_clusters = cluster_hard[cluster_hard > 0.5].sort_values(ascending=False)

    patterns["high_hard_clusters"] = {
        "description": "Clusters with >50% hard examples",
        "clusters": high_hard_clusters.to_dict(),
    }

    # Pattern 3: Clusters with low correctness
    cluster_correctness = merged_df.groupby("cluster")["correctness"].mean()
    low_correctness_clusters = cluster_correctness[
        cluster_correctness < 0.3
    ].sort_values()

    patterns["low_correctness_clusters"] = {
        "description": "Clusters with <30% correctness",
        "clusters": low_correctness_clusters.to_dict(),
    }

    # Pattern 4: Find most problematic cluster (high hard/ambiguous, low correctness)
    cluster_problem_score = (
        cluster_hard * 0.4 + cluster_ambiguous * 0.4 + (1 - cluster_correctness) * 0.2
    )
    most_problematic = cluster_problem_score.sort_values(ascending=False).head(3)

    patterns["most_problematic_clusters"] = {
        "description": "Clusters that appear most problematic (weighted score)",
        "clusters": most_problematic.to_dict(),
    }

    # Pattern 5: Cluster diversity (how many categories each cluster spans)
    cluster_diversity = merged_df.groupby("cluster")["category"].nunique()
    diverse_clusters = cluster_diversity[cluster_diversity >= 3].sort_values(
        ascending=False
    )

    patterns["diverse_clusters"] = {
        "description": "Clusters spanning all 3 cartography categories",
        "clusters": diverse_clusters.to_dict() if len(diverse_clusters) > 0 else {},
    }

    return patterns


def create_cartography_plots(cartography_df: pd.DataFrame, output_dir: str):
    """Create cartography-specific visualizations."""

    # Distribution histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ["confidence", "variability", "correctness"]
    colors = ["blue", "orange", "green"]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.hist(
            cartography_df[metric], bins=50, color=color, alpha=0.7, edgecolor="black"
        )
        ax.set_xlabel(metric.capitalize(), fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Distribution of {metric.capitalize()}", fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

        mean_val = cartography_df[metric].mean()
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "metric_distributions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"   - Saved metric distributions to {fig_path}")
    plt.close()

    # Confidence vs Correctness
    fig, ax = plt.subplots(figsize=(10, 6))

    for category in cartography_df["category"].unique():
        cat_data = cartography_df[cartography_df["category"] == category]
        ax.scatter(
            cat_data["confidence"],
            cat_data["correctness"],
            alpha=0.5,
            s=30,
            label=category,
        )

    ax.set_xlabel("Confidence (Mean Probability)", fontsize=12)
    ax.set_ylabel("Correctness (Fraction Correct)", fontsize=12)
    ax.set_title(
        "Confidence vs Correctness by Category", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    fig_path = os.path.join(output_dir, "confidence_vs_correctness.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"   - Saved confidence vs correctness to {fig_path}")
    plt.close()


def create_integrated_visualizations(merged_df: pd.DataFrame, output_dir: str):
    """Create integrated visualizations showing both cartography and clustering."""

    # Plot 1: Scatter plot with cluster coloring and category markers
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Colored by cluster, markers by category
    ax = axes[0]
    category_markers = {"easy": "o", "hard": "s", "ambiguous": "^"}
    unique_clusters = sorted(merged_df["cluster"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = merged_df[merged_df["cluster"] == cluster_id]

        for category, marker in category_markers.items():
            cat_cluster_data = cluster_data[cluster_data["category"] == category]
            if len(cat_cluster_data) > 0:
                ax.scatter(
                    cat_cluster_data["dim1"],
                    cat_cluster_data["dim2"],
                    c=[colors[i]],
                    marker=marker,
                    s=40,
                    alpha=0.6,
                    label=f"C{cluster_id}-{category}"
                    if len(unique_clusters) <= 5
                    else None,
                )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title("Clusters with Cartography Categories", fontsize=14, fontweight="bold")
    if len(unique_clusters) <= 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: Colored by category
    ax = axes[1]
    category_colors = {"easy": "green", "hard": "red", "ambiguous": "orange"}

    for category, color in category_colors.items():
        cat_data = merged_df[merged_df["category"] == category]
        ax.scatter(
            cat_data["dim1"],
            cat_data["dim2"],
            c=color,
            alpha=0.6,
            s=30,
            label=category,
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(
        "Cartography Categories in Embedding Space", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "integrated_scatter.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"   - Saved integrated scatter plot to {fig_path}")
    plt.close()

    # Plot 2: Heatmap of category distribution across clusters
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create pivot table for heatmap
    pivot = (
        pd.crosstab(merged_df["cluster"], merged_df["category"], normalize="index")
        * 100
    )

    # Reorder columns
    if all(cat in pivot.columns for cat in ["easy", "hard", "ambiguous"]):
        pivot = pivot[["easy", "hard", "ambiguous"]]

    # Create heatmap
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticklabels([f"Cluster {i}" for i in pivot.index], fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)", fontsize=11)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(
                j,
                i,
                f"{pivot.values[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_xlabel("Cartography Category", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title(
        "Category Distribution Across Clusters", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "category_cluster_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"   - Saved category-cluster heatmap to {fig_path}")
    plt.close()

    # Plot 3: Box plots of cartography metrics by cluster
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["confidence", "variability", "correctness"]

    for ax, metric in zip(axes, metrics):
        # Prepare data for box plot
        cluster_ids = sorted(merged_df["cluster"].unique())
        data = [
            merged_df[merged_df["cluster"] == c][metric].values for c in cluster_ids
        ]

        bp = ax.boxplot(
            data, tick_labels=[f"C{i}" for i in cluster_ids], patch_artist=True
        )

        # Color boxes
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(
            f"{metric.capitalize()} Distribution by Cluster",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x labels if many clusters
        if len(cluster_ids) > 10:
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "metrics_by_cluster_boxplot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"   - Saved metrics by cluster boxplot to {fig_path}")
    plt.close()


def export_unified_analysis(
    cartography_dir: str,
    cluster_dir: str,
    dataset_name: str,
    split: str,
    output_dir: str,
    include_rules: bool = True,
):
    """
    Export a unified CSV combining cartography, clustering, and rule-based analysis.

    Parameters:
    -----------
    cartography_dir : str
        Directory containing cartography metrics
    cluster_dir : str
        Directory containing cluster assignments
    dataset_name : str
        Dataset name to load original examples
    split : str
        Dataset split to use
    output_dir : str
        Directory to save outputs
    include_rules : bool
        Whether to include rule-based error detection (default: True)

    Returns:
    --------
    pd.DataFrame : The unified analysis dataframe
    """
    print("=" * 70)
    print("UNIFIED ANALYSIS EXPORT")
    print("=" * 70)

    # Load cartography metrics
    print(f"\n1. Loading cartography metrics from {cartography_dir}...")
    cartography_df = load_cartography_metrics(cartography_dir)
    cartography_df = categorize_examples(cartography_df)
    cartography_df = cartography_df.reset_index()  # Make id a column
    print(f"   Loaded metrics for {len(cartography_df)} examples")

    # Load cluster assignments
    print(f"\n2. Loading cluster assignments from {cluster_dir}...")
    cluster_df = load_cluster_assignments(cluster_dir)
    cluster_df = cluster_df.reset_index()  # Make id a column
    print(f"   Loaded {len(cluster_df)} cluster assignments")

    # Load original dataset
    print(f"\n3. Loading original dataset: {dataset_name}...")
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=dataset_name)
        dataset_split = dataset["train"]
    else:
        dataset = datasets.load_dataset(dataset_name)
        dataset_split = dataset[split]

    dataset_df = pd.DataFrame(dataset_split)

    # Generate IDs if missing
    if "id" not in dataset_df.columns:
        print("   Generating IDs for dataset examples...")
        dataset_with_ids = dataset_split.map(generate_hash_ids)
        dataset_df = pd.DataFrame(dataset_with_ids)

    print(f"   Loaded {len(dataset_df)} examples")

    # Start with dataset as base
    unified_df = dataset_df.copy()

    # Merge cartography metrics
    print("\n4. Merging cartography metrics...")
    unified_df = unified_df.merge(
        cartography_df, on="id", how="left", suffixes=("", "_cart")
    )
    cart_merged = unified_df["category"].notna().sum()
    print(f"   Merged {cart_merged} examples with cartography data")

    # Merge cluster assignments
    print("\n5. Merging cluster assignments...")
    unified_df = unified_df.merge(
        cluster_df, on="id", how="left", suffixes=("", "_cluster")
    )
    cluster_merged = unified_df["cluster"].notna().sum()
    print(f"   Merged {cluster_merged} examples with cluster data")

    # Apply rule-based error detection
    if include_rules:
        print("\n6. Applying rule-based error detection...")
        rule_df = apply_rules_to_dataset(dataset_df)
        unified_df = unified_df.merge(rule_df, on="id", how="left")
        rule_cols = [c for c in rule_df.columns if c.startswith("rule")]
        print(f"   Added {len(rule_cols)} rule-based error flags")

    # Save unified export
    output_file = os.path.join(output_dir, "unified_analysis.csv")
    print(f"\n7. Saving unified analysis to {output_file}...")
    unified_df.to_csv(output_file, index=False)
    print(f"   Saved {len(unified_df)} examples with {len(unified_df.columns)} columns")

    # Create summary statistics
    print("\n8. Computing summary statistics...")
    summary_stats = compute_overlap_statistics(unified_df, include_rules)

    summary_file = os.path.join(output_dir, "overlap_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print(f"   Saved summary statistics to {summary_file}")

    # Print key statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal examples: {len(unified_df)}")
    print(
        f"With cartography data: {cart_merged} ({100 * cart_merged / len(unified_df):.1f}%)"
    )
    print(
        f"With cluster data: {cluster_merged} ({100 * cluster_merged / len(unified_df):.1f}%)"
    )

    if "category" in unified_df.columns:
        print("\nCartography category distribution:")
        cat_counts = unified_df["category"].value_counts()
        for cat, count in cat_counts.items():
            print(f"  - {cat:12s}: {count:6d} ({100 * count / cart_merged:.1f}%)")

    if "cluster" in unified_df.columns:
        n_clusters = unified_df["cluster"].nunique() - (
            1 if -1 in unified_df["cluster"].values else 0
        )
        n_noise = (unified_df["cluster"] == -1).sum()
        print(f"\nClusters found: {n_clusters}")
        print(f"Noise examples: {n_noise} ({100 * n_noise / cluster_merged:.1f}%)")

    if include_rules and "dataset_error_score" in unified_df.columns:
        print("\nRule-based error detection:")
        error_count = unified_df["is_dataset_error"].sum()
        print(
            f"  Flagged as errors: {error_count} ({100 * error_count / len(unified_df):.1f}%)"
        )
        print(f"  Mean error score: {unified_df['dataset_error_score'].mean():.2f}")

    print("\n" + "=" * 70)
    print("UNIFIED EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"- Unified analysis: {output_file}")
    print(f"- Summary statistics: {summary_file}")
    print("=" * 70 + "\n")

    return unified_df


def compute_overlap_statistics(unified_df: pd.DataFrame, include_rules: bool = True):
    """Compute detailed overlap statistics between different analysis dimensions."""

    stats = {
        "total_examples": int(len(unified_df)),
        "data_coverage": {},
        "category_cluster_overlap": {},
        "rule_overlap": {},
        "high_overlap_regions": [],
    }

    # Data coverage
    if "category" in unified_df.columns:
        stats["data_coverage"]["cartography"] = int(
            unified_df["category"].notna().sum()
        )
    if "cluster" in unified_df.columns:
        stats["data_coverage"]["clustering"] = int(unified_df["cluster"].notna().sum())

    # Category-cluster overlap
    if "category" in unified_df.columns and "cluster" in unified_df.columns:
        overlap_df = unified_df[
            unified_df["category"].notna() & unified_df["cluster"].notna()
        ]

        # Distribution by category and cluster
        for category in ["easy", "hard", "ambiguous"]:
            cat_data = overlap_df[overlap_df["category"] == category]
            if len(cat_data) > 0:
                cluster_dist = cat_data["cluster"].value_counts().to_dict()
                stats["category_cluster_overlap"][category] = {
                    "total": int(len(cat_data)),
                    "cluster_distribution": {
                        int(k): int(v) for k, v in cluster_dist.items()
                    },
                }

    # Rule-based overlap
    if include_rules and "dataset_error_score" in unified_df.columns:
        rule_cols = [
            c
            for c in unified_df.columns
            if c.startswith("rule")
            and c not in ["dataset_error_score", "is_dataset_error"]
        ]

        # Overall rule statistics
        stats["rule_overlap"]["total_flagged"] = int(
            unified_df["is_dataset_error"].sum()
        )
        stats["rule_overlap"]["mean_score"] = float(
            unified_df["dataset_error_score"].mean()
        )

        # Rule triggers by category
        if "category" in unified_df.columns:
            stats["rule_overlap"]["by_category"] = {}
            for category in ["easy", "hard", "ambiguous"]:
                cat_data = unified_df[unified_df["category"] == category]
                if len(cat_data) > 0:
                    stats["rule_overlap"]["by_category"][category] = {
                        "flagged_count": int(cat_data["is_dataset_error"].sum()),
                        "flagged_percent": float(
                            100 * cat_data["is_dataset_error"].mean()
                        ),
                        "mean_score": float(cat_data["dataset_error_score"].mean()),
                    }

        # Rule triggers by cluster
        if "cluster" in unified_df.columns:
            stats["rule_overlap"]["by_cluster"] = {}
            cluster_groups = unified_df.groupby("cluster")
            for cluster_id, group in cluster_groups:
                if (
                    cluster_id != -1 and len(group) >= 10
                ):  # Skip noise and small clusters
                    stats["rule_overlap"]["by_cluster"][int(cluster_id)] = {
                        "flagged_count": int(group["is_dataset_error"].sum()),
                        "flagged_percent": float(
                            100 * group["is_dataset_error"].mean()
                        ),
                        "mean_score": float(group["dataset_error_score"].mean()),
                    }

        # Individual rule statistics
        stats["rule_overlap"]["individual_rules"] = {}
        for rule_col in rule_cols:
            triggered = unified_df[rule_col].sum()
            stats["rule_overlap"]["individual_rules"][rule_col] = {
                "triggered_count": int(triggered),
                "triggered_percent": float(100 * triggered / len(unified_df)),
            }

    # Identify high overlap regions (problematic areas)
    if all(
        c in unified_df.columns for c in ["category", "cluster", "is_dataset_error"]
    ):
        # Find category + cluster combinations with high error rates
        overlap_df = unified_df[
            unified_df["category"].notna() & unified_df["cluster"].notna()
        ]

        if len(overlap_df) > 0:
            grouped = (
                overlap_df.groupby(["category", "cluster"])
                .agg({"is_dataset_error": ["sum", "mean", "count"]})
                .reset_index()
            )

            grouped.columns = [
                "category",
                "cluster",
                "error_count",
                "error_rate",
                "total_count",
            ]

            # Filter for significant groups (at least 10 examples, error rate > 30%)
            significant = grouped[
                (grouped["total_count"] >= 10) & (grouped["error_rate"] > 0.3)
            ].sort_values("error_rate", ascending=False)

            for _, row in significant.head(10).iterrows():
                stats["high_overlap_regions"].append(
                    {
                        "category": row["category"],
                        "cluster": int(row["cluster"]),
                        "total_examples": int(row["total_count"]),
                        "error_count": int(row["error_count"]),
                        "error_rate": float(row["error_rate"]),
                    }
                )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Unified dataset analysis (cartography and/or clustering)"
    )
    parser.add_argument(
        "--cartography_dir",
        type=str,
        default=None,
        help="Directory containing cartography output files",
    )
    parser.add_argument(
        "--cluster_dir",
        type=str,
        default=None,
        help="Directory containing clustering output files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Eladio/emrqa-msquad",
        help="Dataset name or path (to get original examples)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional: Dataset split to analyze (required if --cluster_dir is not provided)",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=10,
        help="Number of examples to show per category",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./full_analysis_output",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--export_unified",
        action="store_true",
        help="Export unified CSV combining cartography, clustering, and rule-based analysis",
    )
    parser.add_argument(
        "--no_rules",
        action="store_true",
        help="Exclude rule-based error detection from unified export",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine what analysis to perform
    has_cartography = args.cartography_dir is not None
    has_clustering = args.cluster_dir is not None

    if not has_cartography and not has_clustering:
        print("Error: Must specify at least one of --cartography_dir or --cluster_dir")
        return

    # Resolve split parameter
    split = args.split
    if split is None:
        # Try to extract from cluster metadata if available
        if has_clustering:
            metadata_file = os.path.join(args.cluster_dir, "cluster_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    cluster_metadata = json.load(f)
                split = cluster_metadata.get("split")
                if split:
                    print(f"Using split '{split}' from cluster metadata")

        # If still None, require user to specify
        if split is None:
            print("Error: --split parameter is required.")
            print("Please specify either 'train' or 'validation'")
            return

    # Run all applicable analyses (non-exclusive)

    # 1. Handle unified export if requested
    if args.export_unified:
        if not (has_cartography and has_clustering):
            print(
                "Error: --export_unified requires both --cartography_dir and --cluster_dir"
            )
            return

        export_unified_analysis(
            cartography_dir=args.cartography_dir,
            cluster_dir=args.cluster_dir,
            dataset_name=args.dataset,
            split=split,
            output_dir=args.output_dir,
            include_rules=not args.no_rules,
        )
        # Continue to other analyses instead of returning

    # 2. Cartography-only analysis
    if has_cartography:
        analyze_cartography_only(
            cartography_dir=args.cartography_dir,
            dataset_name=args.dataset,
            split=split,
            n_examples=args.n_examples,
            output_dir=args.output_dir,
        )

    # 3. Clustering-only analysis
    if has_clustering:
        analyze_clustering_only(
            cluster_dir=args.cluster_dir,
            output_dir=args.output_dir,
        )

    # 4. Integrated analysis (if both available)
    if has_cartography and has_clustering:
        analyze_integrated(
            cartography_dir=args.cartography_dir,
            cluster_dir=args.cluster_dir,
            dataset_name=args.dataset,
            split=split,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

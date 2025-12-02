"""
Analysis script for dataset cartography results.

Usage:
    python analyze_cartography.py --cartography_dir ./cartography_output --dataset Eladio/emrqa-msquad
"""

import argparse
import json
import os

import datasets
import matplotlib.pyplot as plt
import pandas as pd

from dataset_cartography import (
    analyze_cartography_by_question_type,
    categorize_examples,
    get_examples_by_category,
    load_cartography_metrics,
)
from helpers import generate_hash_ids


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset cartography results")
    parser.add_argument(
        "--cartography_dir",
        type=str,
        default="./cartography_output",
        help="Directory containing cartography output files",
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
        default="train",
        help="Dataset split to analyze",
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
        default="./cartography_analysis",
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DATASET CARTOGRAPHY ANALYSIS")
    print("=" * 70)

    # Load cartography metrics
    print(f"\n1. Loading cartography metrics from {args.cartography_dir}...")
    cartography_df = load_cartography_metrics(args.cartography_dir)
    print(f"   Loaded metrics for {len(cartography_df)} examples")

    # Categorize examples
    print("\n2. Categorizing examples...")
    cartography_df = categorize_examples(cartography_df)

    category_counts = cartography_df["category"].value_counts()
    print("\n   Category distribution:")
    for cat, count in category_counts.items():
        print(f"   - {cat:15s}: {count:6d} ({100 * count / len(cartography_df):5.1f}%)")

    # Load original dataset
    print(f"\n3. Loading original dataset: {args.dataset}...")
    if args.dataset.endswith(".json") or args.dataset.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=args.dataset)
        dataset_split = dataset["train"]
    else:
        dataset = datasets.load_dataset(args.dataset)
        dataset_split = dataset[args.split]

    dataset_df = pd.DataFrame(dataset_split)

    # Generate IDs if missing (same as in run.py)
    if "id" not in dataset_df.columns:
        print("   Generating IDs for dataset examples...")
        # Apply hash generation to create IDs
        dataset_with_ids = dataset_split.map(generate_hash_ids)
        dataset_df = pd.DataFrame(dataset_with_ids)

    print(f"   Loaded {len(dataset_df)} examples from {args.split} split")

    # Analyze by question type
    print("\n4. Analyzing metrics by question type...")
    question_type_analysis = analyze_cartography_by_question_type(
        cartography_df, dataset_df
    )
    print("\n" + str(question_type_analysis))

    # Save question type analysis
    qtype_file = os.path.join(args.output_dir, "question_type_analysis.csv")
    question_type_analysis.to_csv(qtype_file)
    print(f"\n   Saved to {qtype_file}")

    # Get example samples from each category
    print(f"\n5. Extracting example samples ({args.n_examples} per category)...")

    samples = {}
    samples_files = {}
    for category in ["easy", "hard", "ambiguous"]:
        example_ids = get_examples_by_category(
            cartography_df, category, n=args.n_examples
        )

        # Get full examples from dataset
        category_examples = []
        for ex_id in example_ids:
            # Find example in dataset
            matching = dataset_df[dataset_df["id"] == ex_id]
            if len(matching) > 0:
                ex = matching.iloc[0].to_dict()
                # Add cartography metrics
                metrics = cartography_df.loc[ex_id].to_dict()
                ex["cartography_metrics"] = metrics
                category_examples.append(ex)

        samples[category] = category_examples
        print(f"   - {category:12s}: {len(category_examples)} examples")
        
        # Save each category to its own file
        category_file = os.path.join(args.output_dir, f"{category}_samples.json")
        with open(category_file, "w") as f:
            json.dump(category_examples, f, indent=2, default=str)
        samples_files[category] = category_file

    print(f"\n   Saved category samples to separate files:")
    for category, filepath in samples_files.items():
        print(f"   - {category}: {filepath}")

    # Create detailed visualizations
    print("\n6. Creating additional visualizations...")
    create_additional_plots(cartography_df, args.output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"- Cartography metrics: {args.cartography_dir}/cartography_metrics.csv")
    print(f"- Question type analysis: {qtype_file}")
    print(f"- Visualizations: {args.output_dir}/*.png")
    print("=" * 70 + "\n")


def create_additional_plots(df, output_dir):
    """Create additional analysis plots."""

    # Plot 1: Distribution histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ["confidence", "variability", "correctness"]
    colors = ["blue", "orange", "green"]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.hist(df[metric], bins=50, color=color, alpha=0.7, edgecolor="black")
        ax.set_xlabel(metric.capitalize(), fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Distribution of {metric.capitalize()}", fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add mean line
        mean_val = df[metric].mean()
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

    # Plot 2: Confidence vs Correctness
    fig, ax = plt.subplots(figsize=(10, 6))

    for category in df["category"].unique():
        cat_data = df[df["category"] == category]
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

    # Plot 3: Category breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    ax = axes[0]
    category_counts = df["category"].value_counts()
    colors_map = {
        "easy": "green",
        "hard": "red",
        "ambiguous": "orange",
        "easy_variable": "lightgreen",
    }
    colors = [colors_map.get(cat, "gray") for cat in category_counts.index]

    ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax.set_title("Example Distribution by Category", fontsize=13, fontweight="bold")

    # Box plot
    ax = axes[1]
    categories = df["category"].unique()
    confidence_by_cat = [
        df[df["category"] == cat]["confidence"].values for cat in categories
    ]

    bp = ax.boxplot(confidence_by_cat, labels=categories, patch_artist=True)
    for patch, cat in zip(bp["boxes"], categories):
        patch.set_facecolor(colors_map.get(cat, "gray"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title("Confidence Distribution by Category", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "category_analysis.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"   - Saved category analysis to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()

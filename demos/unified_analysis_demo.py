"""
Example script demonstrating unified analysis export and analysis.

This script shows how to:
1. Export unified analysis combining cartography, clustering, and rule-based detection
2. Analyze the results to find problematic regions
3. Generate summary statistics
"""

# Making sure this can be run from outside the directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_unified_analysis(output_dir="./unified_export"):
    """Load the unified analysis CSV and summary JSON."""

    df = pd.read_csv(f"{output_dir}/unified_analysis.csv")

    with open(f"{output_dir}/overlap_summary.json", "r") as f:
        summary = json.load(f)

    return df, summary


def analyze_high_overlap_regions(df):
    """Find and analyze regions with high overlap between all detection methods."""

    print("\n" + "=" * 70)
    print("HIGH OVERLAP REGIONS ANALYSIS")
    print("=" * 70)

    # Filter for examples with complete data
    complete_df = df[
        df["category"].notna() & df["cluster"].notna() & df["is_dataset_error"].notna()
    ].copy()

    # Group by category and cluster
    grouped = (
        complete_df.groupby(["category", "cluster"])
        .agg(
            {
                "is_dataset_error": ["sum", "mean", "count"],
                "dataset_error_score": "mean",
                "confidence": "mean",
                "correctness": "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    grouped.columns = [
        "category",
        "cluster",
        "error_count",
        "error_rate",
        "total_count",
        "avg_error_score",
        "avg_confidence",
        "avg_correctness",
    ]

    # Filter for significant regions (>= 10 examples, error rate > 30%)
    high_overlap = grouped[
        (grouped["total_count"] >= 10) & (grouped["error_rate"] > 0.3)
    ].sort_values("error_rate", ascending=False)

    print(f"\nFound {len(high_overlap)} high-overlap regions:")
    print("\nTop 10 most problematic regions:")
    print(high_overlap.head(10).to_string(index=False))

    return high_overlap


def analyze_rule_triggers(df):
    """Analyze which rules trigger most frequently in different contexts."""

    print("\n" + "=" * 70)
    print("RULE TRIGGER ANALYSIS")
    print("=" * 70)

    rule_cols = [
        c
        for c in df.columns
        if c.startswith("rule") and c not in ["dataset_error_score", "is_dataset_error"]
    ]

    # Overall rule statistics
    print("\nOverall rule trigger rates:")
    overall_rates = df[rule_cols].mean().sort_values(ascending=False) * 100
    for rule, rate in overall_rates.items():
        print(f"  {rule:40s}: {rate:5.2f}%")

    # By cartography category
    if "category" in df.columns:
        print("\n\nRule triggers by cartography category:")
        for category in ["easy", "hard", "ambiguous"]:
            cat_df = df[df["category"] == category]
            if len(cat_df) > 0:
                print(f"\n  {category.upper()}:")
                cat_rates = cat_df[rule_cols].mean().sort_values(ascending=False) * 100
                for rule, rate in cat_rates.head(5).items():
                    print(f"    {rule:38s}: {rate:5.2f}%")


def find_agreement_disagreement(df):
    """Find examples where methods agree vs disagree."""

    print("\n" + "=" * 70)
    print("METHOD AGREEMENT ANALYSIS")
    print("=" * 70)

    # Complete data only
    complete_df = df[df["category"].notna() & df["is_dataset_error"].notna()].copy()

    # Define "flagged" for each method
    cart_flagged = complete_df["category"] == "ambiguous"
    rule_flagged = complete_df["is_dataset_error"]

    # Agreement cases
    both_flagged = cart_flagged & rule_flagged
    both_clean = ~cart_flagged & ~rule_flagged
    cart_only = cart_flagged & ~rule_flagged
    rule_only = ~cart_flagged & rule_flagged

    print("\nAgreement between cartography and rule-based detection:")
    print(
        f"  Both flag as problematic:  {both_flagged.sum():6d} ({100 * both_flagged.mean():.2f}%)"
    )
    print(
        f"  Both mark as clean:        {both_clean.sum():6d} ({100 * both_clean.mean():.2f}%)"
    )
    print(
        f"  Cartography only:          {cart_only.sum():6d} ({100 * cart_only.mean():.2f}%)"
    )
    print(
        f"  Rules only:                {rule_only.sum():6d} ({100 * rule_only.mean():.2f}%)"
    )

    # Cohen's Kappa for agreement
    agreement = both_flagged.sum() + both_clean.sum()
    total = len(complete_df)
    observed_agreement = agreement / total

    p_yes = (cart_flagged.sum() * rule_flagged.sum()) / (total * total)
    p_no = ((~cart_flagged).sum() * (~rule_flagged).sum()) / (total * total)
    expected_agreement = p_yes + p_no

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    print(f"\nCohen's Kappa (agreement): {kappa:.3f}")

    return {
        "both_flagged": complete_df[both_flagged],
        "both_clean": complete_df[both_clean],
        "cart_only": complete_df[cart_only],
        "rule_only": complete_df[rule_only],
    }


def create_visualizations(df, output_dir="./unified_export"):
    """Create visualizations of the unified analysis."""

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Filter complete data
    complete_df = df[
        df["category"].notna() & df["cluster"].notna() & df["is_dataset_error"].notna()
    ].copy()

    # Visualization 1: Error rate heatmap by category and cluster
    print("\n1. Creating error rate heatmap...")

    # Only use non-noise clusters with enough examples
    cluster_counts = complete_df["cluster"].value_counts()
    valid_clusters = cluster_counts[
        (cluster_counts.index != -1) & (cluster_counts >= 20)
    ].index

    heatmap_df = complete_df[complete_df["cluster"].isin(valid_clusters)]

    if len(heatmap_df) > 0:
        pivot = heatmap_df.pivot_table(
            values="is_dataset_error",
            index="cluster",
            columns="category",
            aggfunc="mean",
        )

        plt.figure(figsize=(10, max(6, len(pivot) * 0.4)))
        sns.heatmap(
            pivot, annot=True, fmt=".2%", cmap="Reds", cbar_kws={"label": "Error Rate"}
        )
        plt.title(
            "Error Rate by Cluster and Cartography Category",
            fontsize=14,
            fontweight="bold",
        )
        plt.ylabel("Cluster ID", fontsize=12)
        plt.xlabel("Cartography Category", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/error_rate_heatmap.png", dpi=300, bbox_inches="tight"
        )
        print(f"   Saved to {output_dir}/error_rate_heatmap.png")
        plt.close()

    # Visualization 2: Scatter plot colored by error status
    print("\n2. Creating scatter plot with error highlighting...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Color by category
    ax = axes[0]
    for category, color in [
        ("easy", "green"),
        ("hard", "orange"),
        ("ambiguous", "red"),
    ]:
        cat_data = complete_df[complete_df["category"] == category]
        ax.scatter(
            cat_data["dim1"], cat_data["dim2"], c=color, alpha=0.4, s=20, label=category
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title("Cartography Categories", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Color by error status
    ax = axes[1]
    clean_data = complete_df[~complete_df["is_dataset_error"]]
    error_data = complete_df[complete_df["is_dataset_error"]]

    ax.scatter(
        clean_data["dim1"], clean_data["dim2"], c="blue", alpha=0.3, s=20, label="Clean"
    )
    ax.scatter(
        error_data["dim1"],
        error_data["dim2"],
        c="red",
        alpha=0.6,
        s=30,
        label="Flagged as Error",
        marker="x",
    )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title("Rule-Based Error Detection", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_scatter.png", dpi=300, bbox_inches="tight")
    print(f"   Saved to {output_dir}/error_scatter.png")
    plt.close()

    # Visualization 3: Rule trigger distribution
    print("\n3. Creating rule trigger distribution plot...")

    rule_cols = [
        c
        for c in df.columns
        if c.startswith("rule") and c not in ["dataset_error_score", "is_dataset_error"]
    ]

    rule_rates = df[rule_cols].mean().sort_values(ascending=False) * 100

    plt.figure(figsize=(12, 6))
    rule_rates.plot(kind="bar", color="steelblue")
    plt.xlabel("Rule", fontsize=12)
    plt.ylabel("Trigger Rate (%)", fontsize=12)
    plt.title(
        "Rule-Based Error Detection Trigger Rates", fontsize=14, fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rule_triggers.png", dpi=300, bbox_inches="tight")
    print(f"   Saved to {output_dir}/rule_triggers.png")
    plt.close()


def export_problematic_examples(df, agreement_data, output_dir="./unified_export"):
    """Export lists of problematic examples for manual review."""

    print("\n" + "=" * 70)
    print("EXPORTING PROBLEMATIC EXAMPLES")
    print("=" * 70)

    # Export examples flagged by both methods
    both_flagged = agreement_data["both_flagged"]
    if len(both_flagged) > 0:
        review_cols = [
            "id",
            "question",
            "context",
            "answer",
            "category",
            "cluster",
            "dataset_error_score",
            "confidence",
            "correctness",
        ]
        review_cols = [c for c in review_cols if c in both_flagged.columns]

        output_file = f"{output_dir}/for_manual_review.csv"
        both_flagged[review_cols].to_csv(output_file, index=False)
        print(f"\n1. Exported {len(both_flagged)} examples flagged by both methods")
        print(f"   Saved to {output_file}")

    # Export high-risk examples (ambiguous + cluster noise + errors)
    high_risk = df[
        (df["category"] == "ambiguous") & (df["cluster"] == -1) & df["is_dataset_error"]
    ]

    if len(high_risk) > 0:
        output_file = f"{output_dir}/high_risk_examples.csv"
        review_cols = [c for c in review_cols if c in high_risk.columns]
        high_risk[review_cols].to_csv(output_file, index=False)
        print(f"\n2. Exported {len(high_risk)} high-risk examples")
        print(f"   Saved to {output_file}")


def main():
    """Main execution function."""

    print("=" * 70)
    print("UNIFIED ANALYSIS EXAMPLE")
    print("=" * 70)
    print("\nThis script demonstrates how to analyze the unified export.")
    print("Make sure you've run the export first:")
    print("  python analyze_dataset.py --export_unified ...")
    print("=" * 70)

    # Load data
    print("\nLoading unified analysis...")
    df, summary = load_unified_analysis()

    print(f"\nLoaded {len(df)} examples")
    print(f"Columns: {len(df.columns)}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS FROM JSON")
    print("=" * 70)
    print(json.dumps(summary, indent=2))

    # Analyze high overlap regions
    high_overlap = analyze_high_overlap_regions(df)
    print(f"\nIdentified {len(high_overlap)} high-overlap regions for investigation.")

    # Analyze rule triggers
    analyze_rule_triggers(df)

    # Find agreement/disagreement
    agreement_data = find_agreement_disagreement(df)

    # Create visualizations
    create_visualizations(df)

    # Export problematic examples
    export_problematic_examples(df, agreement_data)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nCheck the unified_export/ directory for all outputs.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""
Demonstration of Cluster-based Filtering

This demo shows how to use the ClusterFilter class to filter datasets
based on cluster assignments from HDBSCAN clustering analysis.

Usage examples:
    1. Keep only specific clusters
    2. Exclude specific clusters
    3. Handle noise examples (cluster=-1)
    4. Filter by cluster probability
"""

# Making sure this can be run from outside the directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import datasets

from dataset_filters import ClusterFilter, apply_filters
from helpers import generate_hash_ids


def demo_basic_cluster_filtering():
    """Example 1: Exclude specific clusters and noise"""
    print("\n" + "=" * 70)
    print("DEMO 1: Exclude Specific Clusters")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_data = dataset["train"]

    # Generate IDs if needed
    if "id" not in train_data.column_names:
        train_data = train_data.map(generate_hash_ids)

    # Exclude clusters 3, 4, and noise (-1)
    filter_obj = ClusterFilter(
        dataset=train_data,
        cluster_path="./cluster_output",
        exclude_clusters=[2, -1],  # Exclude these clusters
    )

    filtered_data = filter_obj.apply()
    print("\nFiltered dataset, excluding clusters 3, 4, and noise")
    print(f"Resulting size: {len(filtered_data)} examples")
    print("=" * 70 + "\n")


def demo_exclude_clusters():
    """Example 2: Keep all clusters including noise"""
    print("\n" + "=" * 70)
    print("DEMO 2: Keep All Clusters Including Noise")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_data = dataset["train"]

    if "id" not in train_data.column_names:
        train_data = train_data.map(generate_hash_ids)

    # Keep all clusters (exclude nothing)
    filter_obj = ClusterFilter(
        dataset=train_data,
        cluster_path="./cluster_output",
        exclude_clusters=[],  # Don't exclude anything
    )

    filtered_data = filter_obj.apply()
    print(f"\nFiltered to {len(filtered_data)} examples (all clusters including noise)")
    print("=" * 70 + "\n")


def demo_noise_filtering():
    """Example 3: Exclude only noise"""
    print("\n" + "=" * 70)
    print("DEMO 3: Exclude Only Noise (Default Behavior)")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_data = dataset["train"]

    if "id" not in train_data.column_names:
        train_data = train_data.map(generate_hash_ids)

    # Exclude only noise (default behavior)
    filter_obj = ClusterFilter(
        dataset=train_data,
        cluster_path="./cluster_output",
        exclude_clusters=[-1],  # Only exclude noise (this is the default)
    )

    filtered_data = filter_obj.apply()
    print(f"\nFiltered to {len(filtered_data)} examples (excluded noise only)")
    print("=" * 70 + "\n")


def demo_probability_filtering():
    """Example 4: Filter by cluster probability"""
    print("\n" + "=" * 70)
    print("DEMO 4: Filter by Cluster Probability")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_data = dataset["train"]

    if "id" not in train_data.column_names:
        train_data = train_data.map(generate_hash_ids)

    # Keep only high-confidence cluster assignments
    filter_obj = ClusterFilter(
        dataset=train_data,
        cluster_path="./cluster_output",
        min_probability=0.8,  # Only examples with 80%+ probability
        exclude_noise=True,
    )

    filtered_data = filter_obj.apply()
    print(
        f"\nFiltered to {len(filtered_data)} examples with cluster probability >= 0.8"
    )
    print("=" * 70 + "\n")


def demo_combined_filtering():
    """Example 5: Combine cluster and cartography filtering"""
    print("\n" + "=" * 70)
    print("DEMO 5: Combined Filtering (Cluster + Cartography)")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_data = dataset["train"]

    if "id" not in train_data.column_names:
        train_data = train_data.map(generate_hash_ids)

    # Apply both filters using apply_filters
    filter_config = {
        "ambiguous": {
            "enabled": True,
            "metrics_path": "./cartography_output",
            "top_fraction": 0.33,
            "apply_rule_based_filter": False,
        },
        "cluster": {
            "enabled": True,
            "cluster_path": "./cluster_output",
            "exclude_clusters": [1, 2, -1],  # Exclude clusters 3, 4, and noise
            "min_probability": 0.7,  # High confidence only
        },
    }

    filtered_data = apply_filters(train_data, filter_config)
    print(f"\nFiltered to {len(filtered_data)} examples using combined filters")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CLUSTER FILTERING DEMONSTRATIONS")
    print("=" * 70)
    print("\nThese examples show various ways to filter datasets using cluster")
    print("assignments from HDBSCAN clustering analysis.")
    print("\nNote: Make sure you have cluster_output/ directory with cluster")
    print("assignments before running these demos.")
    print("=" * 70)

    # Run demos (comment out the ones you don't want to run)
    # demo_basic_cluster_filtering()
    # demo_exclude_clusters()
    # demo_noise_filtering()
    # demo_probability_filtering()
    # demo_combined_filtering()

    print("\nTo run these demos, uncomment the desired demo_* calls at the bottom")
    print("of this file.\n")

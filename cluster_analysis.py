"""
Clustering Analysis for QA Dataset Embeddings

This module performs clustering on dataset embeddings to identify
semantic regions and patterns in the data using HDBSCAN.

Pipeline:
1. Normalize embeddings
2. Reduce to intermediate dimensions (30-50) using PCA
3. Cluster using HDBSCAN (Hierarchical DBSCAN)
4. Further reduce to 2D for visualization
"""

import argparse
import json
import os
import warnings
from typing import Optional

import datasets
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from extract_embeddings import load_embeddings
from helpers import generate_hash_ids

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

#! cluster_sample.json showing representative samples of each cluster needs to be updated to also show the answers.


def load_cluster_assignments(cluster_path: str) -> pd.DataFrame:
    """
    Load cluster assignments from file or directory.

    Args:
        cluster_path: Path to cluster output directory or CSV file

    Returns:
        DataFrame with cluster assignments (indexed by example_id)
    """
    if os.path.isdir(cluster_path):
        # Load from cluster_assignments.csv in the directory
        csv_path = os.path.join(cluster_path, "cluster_assignments.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"cluster_assignments.csv not found in {cluster_path}"
            )
        cluster_df = pd.read_csv(csv_path, index_col="id")
    elif os.path.isfile(cluster_path):
        # Load CSV directly
        cluster_df = pd.read_csv(cluster_path, index_col="id")
    else:
        raise FileNotFoundError(f"Cluster assignments not found: {cluster_path}")

    return cluster_df


class ClusterAnalyzer:
    """
    Performs clustering analysis on embeddings using HDBSCAN.

    Pipeline:
    1. Normalize embeddings
    2. Reduce to intermediate dimensions (30-50) using PCA
    3. Cluster using HDBSCAN
    4. Reduce to 2D for visualization using UMAP
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        normalize: bool = True,
    ):
        """
        Initialize the cluster analyzer.

        Args:
            embeddings: Embedding vectors (n_samples, n_features)
            normalize: Whether to normalize embeddings before clustering
        """
        self.embeddings = embeddings
        self.n_samples = embeddings.shape[0]
        self.n_features = embeddings.shape[1]

        # Normalize embeddings
        if normalize:
            self.scaler = StandardScaler()
            self.embeddings_normalized = self.scaler.fit_transform(embeddings)
        else:
            self.scaler = None
            self.embeddings_normalized = embeddings

        self.embeddings_reduced = None
        self.cluster_labels = None
        self.clusterer = None
        self.reduction_dim = None

    def reduce_for_clustering(
        self,
        n_components: int = 50,
        method: str = "pca",
        random_state: int = 42,
        **kwargs,
    ) -> np.ndarray:
        """
        Reduce dimensionality before clustering.

        Args:
            n_components: Number of dimensions to reduce to (30-50 recommended)
            method: Reduction method ('pca' or 'umap')
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for the reducer

        Returns:
            Reduced embeddings
        """
        print(
            f"\nReducing dimensions with {method.upper()} to {n_components}D for clustering..."
        )

        if method == "pca":
            reducer = PCA(
                n_components=n_components, random_state=random_state, **kwargs
            )
            self.embeddings_reduced = reducer.fit_transform(self.embeddings_normalized)
            variance_explained = sum(reducer.explained_variance_ratio_)
            print(f"Variance explained: {variance_explained:.2%}")

        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components, random_state=random_state, **kwargs
            )
            self.embeddings_reduced = reducer.fit_transform(self.embeddings_normalized)

        else:
            raise ValueError(
                f"Unknown reduction method: {method}. Use 'pca' or 'umap'."
            )

        self.reduction_dim = n_components
        print(f"Reduction complete! Shape: {self.embeddings_reduced.shape}")
        return self.embeddings_reduced

    def cluster_hdbscan(
        self,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform HDBSCAN clustering on reduced embeddings.

        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Conservative estimate of number of samples in a neighborhood.
                        Defaults to min_cluster_size if not specified.
            metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
            cluster_selection_epsilon: Distance threshold for cluster merging
            **kwargs: Additional arguments for HDBSCAN

        Returns:
            Array of cluster labels (-1 indicates noise)
        """
        if self.embeddings_reduced is None:
            raise ValueError(
                "Must call reduce_for_clustering() before clustering. "
                "Use reduce_for_clustering() to reduce to 30-50 dimensions first."
            )

        print("\nPerforming HDBSCAN clustering...")
        print(f"  min_cluster_size={min_cluster_size}")

        # Use min_cluster_size as default if min_samples not specified
        if min_samples is None:
            min_samples = min_cluster_size

        print(f"  min_samples={min_samples}")
        print(f"  metric={metric}")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            algorithm="best",  # Uses faster approximations for large datasets
            **kwargs,
        )

        self.cluster_labels = self.clusterer.fit_predict(self.embeddings_reduced)

        # Compute statistics
        n_clusters = len(set(self.cluster_labels)) - (
            1 if -1 in self.cluster_labels else 0
        )
        n_noise = list(self.cluster_labels).count(-1)
        noise_percentage = 100 * n_noise / self.n_samples

        print("\nClustering complete!")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise} ({noise_percentage:.1f}%)")

        if n_clusters > 0:
            cluster_sizes = np.bincount(self.cluster_labels[self.cluster_labels >= 0])
            print(f"Cluster sizes: {cluster_sizes}")
            print(f"  Mean: {cluster_sizes.mean():.1f}")
            print(f"  Median: {np.median(cluster_sizes):.1f}")
            print(f"  Min: {cluster_sizes.min()}")
            print(f"  Max: {cluster_sizes.max()}")

        return self.cluster_labels

    def reduce_for_visualization(
        self,
        method: str = "umap",
        random_state: int = 42,
        **kwargs,
    ) -> np.ndarray:
        """
        Reduce to 2D for visualization.

        Args:
            method: Reduction method ('pca', 'umap')
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for the reducer

        Returns:
            2D reduced embeddings
        """
        print(f"\nReducing to 2D for visualization using {method.upper()}...")

        # Use the clustered embeddings if available, otherwise use normalized
        embeddings_to_reduce = (
            self.embeddings_reduced
            if self.embeddings_reduced is not None
            else self.embeddings_normalized
        )

        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state, **kwargs)
            reduced_2d = reducer.fit_transform(embeddings_to_reduce)
            variance_explained = sum(reducer.explained_variance_ratio_)
            print(f"Variance explained: {variance_explained:.2%}")

        elif method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=random_state, **kwargs)
            reduced_2d = reducer.fit_transform(embeddings_to_reduce)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'.")

        print(f"Reduction complete! Shape: {reduced_2d.shape}")
        return reduced_2d

    def get_cluster_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for each cluster.

        Returns:
            DataFrame with cluster statistics
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet")

        unique_labels = np.unique(self.cluster_labels)
        stats = []

        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_embeddings = self.embeddings_normalized[mask]

            # Additional HDBSCAN-specific metrics
            if self.clusterer is not None and label >= 0:
                # Get cluster persistence (strength)
                cluster_persistence = self.clusterer.cluster_persistence_[label]
            else:
                cluster_persistence = None

            stats.append(
                {
                    "cluster": int(label),
                    "size": int(mask.sum()),
                    "percentage": float(100 * mask.sum() / self.n_samples),
                    "mean_norm": float(np.linalg.norm(cluster_embeddings.mean(axis=0))),
                    "std_norm": float(np.linalg.norm(cluster_embeddings.std(axis=0))),
                    "persistence": float(cluster_persistence)
                    if cluster_persistence is not None
                    else None,
                }
            )

        df = pd.DataFrame(stats)
        df = df.sort_values("size", ascending=False)
        return df

    def get_cluster_probabilities(self) -> Optional[np.ndarray]:
        """
        Get HDBSCAN cluster membership probabilities.

        Returns:
            Array of probabilities (or None if not available)
        """
        if self.clusterer is None:
            return None
        return self.clusterer.probabilities_


def analyze_clusters(
    embedding_dir: str,
    output_dir: str,
    dataset_name: str = "Eladio/emrqa-msquad",
    split: str = "train",
    max_samples: Optional[int] = None,
    reduction_dim: int = 50,
    reduction_method: str = "pca",
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    visualization_method: str = "umap",
):
    """
    Perform complete clustering analysis on embeddings using HDBSCAN.

    Args:
        embedding_dir: Directory containing embeddings
        output_dir: Directory to save results
        dataset_name: Dataset name (to load original examples)
        split: Dataset split
        max_samples: Maximum number of samples to cluster (None = all)
        reduction_dim: Intermediate dimensionality (30-50 recommended)
        reduction_method: Method for initial reduction ('pca' or 'umap')
        min_cluster_size: Minimum size for HDBSCAN clusters
        min_samples: Minimum samples for HDBSCAN (defaults to min_cluster_size)
        metric: Distance metric for HDBSCAN
        visualization_method: Method for 2D visualization ('umap' or 'pca')
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HDBSCAN CLUSTERING ANALYSIS")
    print("=" * 70)

    # Load embeddings
    print(f"\n1. Loading embeddings from {embedding_dir}...")
    embeddings, metadata = load_embeddings(embedding_dir)

    # Sample if requested
    if max_samples and max_samples < len(embeddings):
        print(f"   Sampling {max_samples} from {embeddings.shape[0]} embeddings...")
        indices = np.random.RandomState(42).choice(
            len(embeddings), max_samples, replace=False
        )
        indices = np.sort(indices)  # Keep order for dataset alignment
        embeddings = embeddings[indices]

    print(
        f"   Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}"
    )
    print("   Embedding type: [CLS + answer span]")

    # Load dataset
    print(f"\n2. Loading dataset: {dataset_name}...")
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=dataset_name)
        data_split = dataset["train"]
    else:
        dataset = datasets.load_dataset(dataset_name)
        data_split = dataset[split]

    # Generate IDs if missing
    if "id" not in data_split.column_names:
        print("   Generating IDs for dataset examples...")
        data_split = data_split.map(generate_hash_ids)

    # Match dataset size with embeddings (including sampling)
    if max_samples and max_samples < len(data_split):
        # Use same indices as embeddings if we sampled
        data_split = data_split.select(
            indices.tolist() if max_samples else range(len(embeddings))
        )
    elif len(data_split) > len(embeddings):
        data_split = data_split.select(range(len(embeddings)))

    dataset_df = pd.DataFrame(data_split)
    print(f"   Loaded {len(dataset_df)} examples")

    # Initialize analyzer
    print("\n3. Initializing cluster analyzer...")
    analyzer = ClusterAnalyzer(embeddings)

    # Reduce dimensions for clustering
    print(f"\n4. Reducing dimensions for clustering ({reduction_dim}D)...")
    analyzer.reduce_for_clustering(
        n_components=reduction_dim,
        method=reduction_method,
    )

    # Perform HDBSCAN clustering
    print("\n5. Performing HDBSCAN clustering...")
    cluster_labels = analyzer.cluster_hdbscan(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )

    # Add cluster labels to dataframe
    dataset_df["cluster"] = cluster_labels

    # Get cluster probabilities
    probabilities = analyzer.get_cluster_probabilities()
    if probabilities is not None:
        dataset_df["cluster_probability"] = probabilities

    # Get cluster statistics
    print("\n6. Computing cluster statistics...")
    cluster_stats = analyzer.get_cluster_statistics()
    print("\nCluster Statistics:")
    print(cluster_stats.to_string(index=False))

    # Save cluster statistics
    stats_file = os.path.join(output_dir, "cluster_statistics.csv")
    cluster_stats.to_csv(stats_file, index=False)
    print(f"\nSaved cluster statistics to {stats_file}")

    # Reduce to 2D for visualization
    print(f"\n7. Reducing to 2D for visualization ({visualization_method.upper()})...")
    reduced_2d = analyzer.reduce_for_visualization(method=visualization_method)

    dataset_df["dim1"] = reduced_2d[:, 0]
    dataset_df["dim2"] = reduced_2d[:, 1]

    # Save results
    print("\n8. Saving results...")

    # Save cluster assignments
    assignments_cols = ["id", "cluster", "dim1", "dim2"]
    if "cluster_probability" in dataset_df.columns:
        assignments_cols.append("cluster_probability")

    assignments_file = os.path.join(output_dir, "cluster_assignments.csv")
    dataset_df[assignments_cols].to_csv(assignments_file, index=False)
    print(f"   Saved cluster assignments to {assignments_file}")

    # Save cluster examples (excluding noise)
    cluster_samples = {}
    for cluster_id in cluster_stats["cluster"].values:
        if cluster_id == -1:
            # Save some noise examples separately
            noise_examples = dataset_df[dataset_df["cluster"] == -1].head(10)
            cluster_samples["noise"] = [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row.get("answers", {}).get("text", [None])[0]
                    if isinstance(row.get("answers"), dict)
                    else row.get("answers"),
                    "context": row["context"][:200] + "..."
                    if len(row["context"]) > 200
                    else row["context"],
                    "probability": float(row.get("cluster_probability", 0.0)),
                }
                for _, row in noise_examples.iterrows()
            ]
            continue

        cluster_examples = dataset_df[dataset_df["cluster"] == cluster_id].head(10)
        cluster_samples[f"cluster_{cluster_id}"] = [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row.get("answers", {}).get("text", [None])[0]
                if isinstance(row.get("answers"), dict)
                else row.get("answers"),
                "context": row["context"][:200] + "..."
                if len(row["context"]) > 200
                else row["context"],
                "probability": float(row.get("cluster_probability", 1.0)),
            }
            for _, row in cluster_examples.iterrows()
        ]

    samples_file = os.path.join(output_dir, "cluster_samples.json")
    with open(samples_file, "w") as f:
        json.dump(cluster_samples, f, indent=2)
    print(f"   Saved cluster samples to {samples_file}")

    # Save metadata
    analysis_metadata = {
        "embedding_dir": embedding_dir,
        "n_samples": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "reduction_dim": reduction_dim,
        "reduction_method": reduction_method,
        "clustering_method": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples or min_cluster_size,
        "metric": metric,
        "n_clusters": int((cluster_stats["cluster"] >= 0).sum()),
        "n_noise": int((cluster_labels == -1).sum()),
        "visualization_method": visualization_method,
    }

    metadata_file = os.path.join(output_dir, "cluster_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(analysis_metadata, f, indent=2)
    print(f"   Saved metadata to {metadata_file}")

    # Create visualizations
    print("\n9. Creating visualizations...")
    create_cluster_visualizations(
        dataset_df,
        cluster_stats,
        output_dir,
        visualization_method,
    )

    print("\n" + "=" * 70)
    print("HDBSCAN CLUSTERING ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"- Cluster statistics: {stats_file}")
    print(f"- Cluster assignments: {assignments_file}")
    print(f"- Cluster samples: {samples_file}")
    print(f"- Visualizations: {output_dir}/*.png")
    print("=" * 70 + "\n")


def create_cluster_visualizations(
    df: pd.DataFrame,
    cluster_stats: pd.DataFrame,
    output_dir: str,
    reduction_method: str,
):
    """Create visualizations for HDBSCAN cluster analysis."""

    # Plot 1: Cluster scatter plot with probabilities
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot colored by cluster
    ax = axes[0]
    unique_clusters = sorted(df["cluster"].unique())

    # Use a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = df[df["cluster"] == cluster_id]
        label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"

        # Size by probability if available
        if "cluster_probability" in df.columns:
            sizes = cluster_data["cluster_probability"] * 50 + 10
        else:
            sizes = 20

        # Use distinct color and styling for noise
        if cluster_id == -1:
            color = "#808080"  # Gray color for noise
            alpha = 0.4
            marker = "x"
            sizes = 15  # Smaller, uniform size for noise
        else:
            color = colors[i]
            alpha = 0.6
            marker = "o"

        ax.scatter(
            cluster_data["dim1"],
            cluster_data["dim2"],
            c=[color],
            label=label,
            alpha=alpha,
            s=sizes,
            marker=marker,
            edgecolors="black" if cluster_id == -1 else "none",
            linewidths=0.5 if cluster_id == -1 else 0,
        )

    ax.set_xlabel(f"{reduction_method.upper()} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{reduction_method.upper()} Dimension 2", fontsize=12)
    ax.set_title("HDBSCAN Cluster Visualization", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bar chart of cluster sizes
    ax = axes[1]
    non_noise = cluster_stats[cluster_stats["cluster"] >= 0].copy()

    if len(non_noise) > 0:
        bars = ax.bar(
            non_noise["cluster"].astype(str),
            non_noise["size"],
            color=colors[: len(non_noise)],
            alpha=0.7,
        )
        ax.set_xlabel("Cluster ID", fontsize=12)
        ax.set_ylabel("Number of Examples", fontsize=12)
        ax.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "cluster_visualization.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"   Saved cluster visualization to {fig_path}")
    plt.close()

    # Plot 2: Cluster persistence (HDBSCAN-specific)
    if "persistence" in cluster_stats.columns and len(non_noise) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out None values
        valid_persistence = non_noise[non_noise["persistence"].notna()].copy()

        if len(valid_persistence) > 0:
            bars = ax.bar(
                valid_persistence["cluster"].astype(str),
                valid_persistence["persistence"],
                color=colors[: len(valid_persistence)],
                alpha=0.7,
            )

            ax.set_xlabel("Cluster ID", fontsize=12)
            ax.set_ylabel("Cluster Persistence (Stability)", fontsize=12)
            ax.set_title(
                "HDBSCAN Cluster Persistence\n(Higher = More Stable)",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            fig_path = os.path.join(output_dir, "cluster_persistence.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"   Saved cluster persistence plot to {fig_path}")
            plt.close()

    # Plot 3: Probability distribution
    if "cluster_probability" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of all probabilities
        ax = axes[0]
        ax.hist(df["cluster_probability"], bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Cluster Membership Probability", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            "Distribution of Cluster Probabilities", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Box plot by cluster
        ax = axes[1]
        cluster_probs = [
            df[df["cluster"] == c]["cluster_probability"].values
            for c in sorted(df["cluster"].unique())
            if c >= 0
        ]

        if cluster_probs:
            ax.boxplot(
                cluster_probs,
                labels=[str(c) for c in sorted(df["cluster"].unique()) if c >= 0],
            )
            ax.set_xlabel("Cluster ID", fontsize=12)
            ax.set_ylabel("Membership Probability", fontsize=12)
            ax.set_title(
                "Probability Distribution by Cluster", fontsize=14, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig_path = os.path.join(output_dir, "cluster_probabilities.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"   Saved probability distribution plot to {fig_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform HDBSCAN clustering analysis on embeddings"
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory containing saved embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cluster_output",
        help="Directory to save clustering results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Eladio/emrqa-msquad",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to analyze",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to cluster (useful for large datasets)",
    )
    parser.add_argument(
        "--reduction_dim",
        type=int,
        default=50,
        help="Intermediate dimensionality for clustering (30-50 recommended)",
    )
    parser.add_argument(
        "--reduction_method",
        type=str,
        default="pca",
        choices=["pca", "umap"],
        help="Method for initial dimensionality reduction",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="Minimum size for HDBSCAN clusters",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="Minimum samples for HDBSCAN (defaults to min_cluster_size)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for HDBSCAN",
    )
    parser.add_argument(
        "--visualization_method",
        type=str,
        default="umap",
        choices=["pca", "umap"],
        help="Method for 2D visualization",
    )

    args = parser.parse_args()

    analyze_clusters(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        reduction_dim=args.reduction_dim,
        reduction_method=args.reduction_method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        visualization_method=args.visualization_method,
    )

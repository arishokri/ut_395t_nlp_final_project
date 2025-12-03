"""
Clustering Analysis for QA Dataset Embeddings

This module performs clustering on dataset embeddings to identify
semantic regions and patterns in the data.
"""

import argparse
import json
import os
from typing import Optional, Tuple

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from extract_embeddings import load_embeddings
from helpers import generate_hash_ids


class ClusterAnalyzer:
    """
    Performs clustering analysis on embeddings.

    Supports:
    - Multiple clustering algorithms (K-means, DBSCAN)
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Cluster quality metrics
    - Visualization
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

        self.cluster_labels = None
        self.cluster_method = None
        self.cluster_params = None

    def cluster_kmeans(
        self,
        n_clusters: int = 10,
        random_state: int = 42,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform K-means clustering.

        Args:
            n_clusters: Number of clusters
            random_state: Random seed
            **kwargs: Additional arguments for KMeans

        Returns:
            Array of cluster labels
        """
        print(f"\nPerforming K-means clustering with {n_clusters} clusters...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            **kwargs,
        )

        self.cluster_labels = kmeans.fit_predict(self.embeddings_normalized)
        self.cluster_method = "kmeans"
        self.cluster_params = {"n_clusters": n_clusters, "random_state": random_state}

        # Compute metrics
        silhouette = silhouette_score(self.embeddings_normalized, self.cluster_labels)

        print("Clustering complete!")
        print(f"Silhouette score: {silhouette:.3f}")
        print(f"Cluster sizes: {np.bincount(self.cluster_labels)}")

        return self.cluster_labels

    def cluster_dbscan(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering.

        Args:
            eps: Maximum distance between samples in a neighborhood
            min_samples: Minimum samples in a neighborhood for a core point
            **kwargs: Additional arguments for DBSCAN

        Returns:
            Array of cluster labels (-1 indicates noise)
        """
        print(
            f"\nPerforming DBSCAN clustering (eps={eps}, min_samples={min_samples})..."
        )

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.cluster_labels = dbscan.fit_predict(self.embeddings_normalized)
        self.cluster_method = "dbscan"
        self.cluster_params = {"eps": eps, "min_samples": min_samples}

        # Compute metrics
        n_clusters = len(set(self.cluster_labels)) - (
            1 if -1 in self.cluster_labels else 0
        )
        n_noise = list(self.cluster_labels).count(-1)

        print("Clustering complete!")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(
            f"Cluster sizes: {np.bincount(self.cluster_labels[self.cluster_labels >= 0])}"
        )

        if n_clusters > 1:
            # Only compute silhouette if we have multiple clusters
            non_noise_mask = self.cluster_labels >= 0
            if non_noise_mask.sum() > 0:
                silhouette = silhouette_score(
                    self.embeddings_normalized[non_noise_mask],
                    self.cluster_labels[non_noise_mask],
                )
                print(f"Silhouette score (excluding noise): {silhouette:.3f}")

        return self.cluster_labels

    def find_optimal_k(
        self,
        k_range: Tuple[int, int] = (2, 20),
        metric: str = "silhouette",
    ) -> int:
        """
        Find optimal number of clusters using elbow method or silhouette score.

        Args:
            k_range: Range of k values to try (min, max)
            metric: Metric to use ('silhouette' or 'inertia')

        Returns:
            Optimal k value
        """
        print(f"\nFinding optimal k in range {k_range}...")

        k_values = range(k_range[0], k_range[1] + 1)
        scores = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings_normalized)

            if metric == "silhouette":
                score = silhouette_score(self.embeddings_normalized, labels)
            elif metric == "inertia":
                score = -kmeans.inertia_  # Negative for consistency (higher is better)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)
            print(f"k={k}: {metric}={score:.3f}")

        # Find best k
        optimal_k = k_values[np.argmax(scores)]
        print(f"\nOptimal k: {optimal_k} (best {metric}: {max(scores):.3f})")

        return optimal_k

    def reduce_dimensions(
        self,
        method: str = "umap",
        n_components: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """
        Reduce dimensionality for visualization.

        Args:
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components (usually 2 for visualization)
            **kwargs: Additional arguments for the reducer

        Returns:
            Reduced embeddings
        """
        print(f"\nReducing dimensions with {method.upper()} to {n_components}D...")

        if method == "pca":
            reducer = PCA(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(self.embeddings_normalized)
            variance_explained = sum(reducer.explained_variance_ratio_)
            print(f"Variance explained: {variance_explained:.2%}")

        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=kwargs.get("random_state", 42),
                **{k: v for k, v in kwargs.items() if k != "random_state"},
            )
            reduced = reducer.fit_transform(self.embeddings_normalized)

        elif method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError(
                    "UMAP not available. Install with: pip install umap-learn"
                )
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=kwargs.get("random_state", 42),
                **{k: v for k, v in kwargs.items() if k != "random_state"},
            )
            reduced = reducer.fit_transform(self.embeddings_normalized)

        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Reduction complete! Shape: {reduced.shape}")
        return reduced

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

            # Compute statistics
            stats.append(
                {
                    "cluster": int(label),
                    "size": int(mask.sum()),
                    "percentage": float(100 * mask.sum() / self.n_samples),
                    "mean_norm": float(np.linalg.norm(cluster_embeddings.mean(axis=0))),
                    "std_norm": float(np.linalg.norm(cluster_embeddings.std(axis=0))),
                }
            )

        df = pd.DataFrame(stats)
        df = df.sort_values("size", ascending=False)
        return df


def analyze_clusters(
    embedding_dir: str,
    output_dir: str,
    dataset_name: str = "Eladio/emrqa-msquad",
    split: str = "train",
    embedding_type: str = "cls",
    clustering_method: str = "kmeans",
    n_clusters: Optional[int] = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    reduction_method: str = "umap",
    find_optimal: bool = False,
):
    """
    Perform complete clustering analysis on embeddings.

    Args:
        embedding_dir: Directory containing embeddings
        output_dir: Directory to save results
        dataset_name: Dataset name (to load original examples)
        split: Dataset split
        embedding_type: Type of embedding to analyze
        clustering_method: Clustering algorithm ('kmeans' or 'dbscan')
        n_clusters: Number of clusters for K-means (auto-detected if None)
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        reduction_method: Dimensionality reduction method
        find_optimal: Whether to find optimal k before clustering
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CLUSTERING ANALYSIS")
    print("=" * 70)

    # Load embeddings
    print(f"\n1. Loading embeddings from {embedding_dir}...")
    embeddings, metadata = load_embeddings(embedding_dir, embedding_type)
    print(
        f"   Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}"
    )

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

    # Match dataset size with embeddings
    if len(data_split) > len(embeddings):
        data_split = data_split.select(range(len(embeddings)))

    dataset_df = pd.DataFrame(data_split)
    print(f"   Loaded {len(dataset_df)} examples")

    # Initialize analyzer
    analyzer = ClusterAnalyzer(embeddings)

    # Find optimal k if requested
    if clustering_method == "kmeans" and (find_optimal or n_clusters is None):
        print("\n3. Finding optimal number of clusters...")
        optimal_k = analyzer.find_optimal_k(k_range=(2, min(20, len(embeddings) // 50)))
        n_clusters = n_clusters or optimal_k
    else:
        print(f"\n3. Using {n_clusters or 'auto'} clusters")

    # Perform clustering
    print("\n4. Performing clustering...")
    if clustering_method == "kmeans":
        if n_clusters is None:
            n_clusters = 10
        cluster_labels = analyzer.cluster_kmeans(n_clusters=n_clusters)
    elif clustering_method == "dbscan":
        cluster_labels = analyzer.cluster_dbscan(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        )
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

    # Add cluster labels to dataframe
    dataset_df["cluster"] = cluster_labels

    # Get cluster statistics
    print("\n5. Computing cluster statistics...")
    cluster_stats = analyzer.get_cluster_statistics()
    print("\nCluster Statistics:")
    print(cluster_stats.to_string(index=False))

    # Save cluster statistics
    stats_file = os.path.join(output_dir, "cluster_statistics.csv")
    cluster_stats.to_csv(stats_file, index=False)
    print(f"\nSaved cluster statistics to {stats_file}")

    # Dimensionality reduction for visualization
    print(f"\n6. Performing dimensionality reduction ({reduction_method.upper()})...")
    reduced_embeddings = analyzer.reduce_dimensions(method=reduction_method)

    dataset_df["dim1"] = reduced_embeddings[:, 0]
    dataset_df["dim2"] = reduced_embeddings[:, 1]

    # Save results
    print("\n7. Saving results...")

    # Save cluster assignments
    assignments_file = os.path.join(output_dir, "cluster_assignments.csv")
    dataset_df[["id", "cluster", "dim1", "dim2"]].to_csv(assignments_file, index=False)
    print(f"   Saved cluster assignments to {assignments_file}")

    # Save cluster examples
    cluster_samples = {}
    for cluster_id in cluster_stats["cluster"].values:
        if cluster_id == -1:
            continue  # Skip noise cluster for now

        cluster_examples = dataset_df[dataset_df["cluster"] == cluster_id].head(10)
        cluster_samples[f"cluster_{cluster_id}"] = [
            {
                "id": row["id"],
                "question": row["question"],
                "context": row["context"][:200] + "..."
                if len(row["context"]) > 200
                else row["context"],
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
        "embedding_type": embedding_type,
        "n_samples": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "clustering_method": clustering_method,
        "clustering_params": analyzer.cluster_params,
        "n_clusters": int(len(cluster_stats)),
        "reduction_method": reduction_method,
    }

    metadata_file = os.path.join(output_dir, "cluster_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(analysis_metadata, f, indent=2)
    print(f"   Saved metadata to {metadata_file}")

    # Create visualizations
    print("\n8. Creating visualizations...")
    create_cluster_visualizations(
        dataset_df,
        cluster_stats,
        output_dir,
        reduction_method,
    )

    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS COMPLETE")
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
    """Create visualizations for cluster analysis."""

    # Plot 1: Cluster scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot colored by cluster
    ax = axes[0]
    unique_clusters = sorted(df["cluster"].unique())

    # Use a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = df[df["cluster"] == cluster_id]
        label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
        ax.scatter(
            cluster_data["dim1"],
            cluster_data["dim2"],
            c=[colors[i]],
            label=label,
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel(f"{reduction_method.upper()} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{reduction_method.upper()} Dimension 2", fontsize=12)
    ax.set_title("Cluster Visualization", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bar chart of cluster sizes
    ax = axes[1]
    non_noise = cluster_stats[cluster_stats["cluster"] >= 0].copy()

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

    # Plot 2: Cluster statistics
    if len(non_noise) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(non_noise))
        width = 0.35

        ax.bar(
            x - width / 2,
            non_noise["mean_norm"],
            width,
            label="Mean Norm",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            non_noise["std_norm"],
            width,
            label="Std Norm",
            alpha=0.7,
        )

        ax.set_xlabel("Cluster ID", fontsize=12)
        ax.set_ylabel("Norm Value", fontsize=12)
        ax.set_title("Cluster Embedding Statistics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(non_noise["cluster"].astype(str))
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig_path = os.path.join(output_dir, "cluster_statistics_plot.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"   Saved cluster statistics plot to {fig_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform clustering analysis on embeddings"
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
        "--embedding_type",
        type=str,
        default="cls",
        help="Type of embedding to analyze",
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "dbscan"],
        help="Clustering algorithm to use",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters (for K-means)",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter",
    )
    parser.add_argument(
        "--reduction_method",
        type=str,
        default="umap",
        choices=["pca", "tsne", "umap"],
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--find_optimal",
        action="store_true",
        help="Find optimal number of clusters",
    )

    args = parser.parse_args()

    analyze_clusters(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        split=args.split,
        embedding_type=args.embedding_type,
        clustering_method=args.clustering_method,
        n_clusters=args.n_clusters,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        reduction_method=args.reduction_method,
        find_optimal=args.find_optimal,
    )

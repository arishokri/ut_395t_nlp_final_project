"""
Dataset Filtering Module

This module provides various strategies for filtering datasets based on
analysis results (e.g., cartography metrics, clustering, embeddings).

Each filtering strategy can be applied independently or combined.
Analysis results should be pre-computed and saved to disk before filtering.
"""

import os
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset

from dataset_cartography import categorize_examples, load_cartography_metrics
from cluster_analysis import load_cluster_assignments


class DatasetFilter:
    """Base class for dataset filtering strategies."""

    def __init__(self, dataset: Dataset):
        """
        Initialize the filter with a dataset.

        Args:
            dataset: HuggingFace Dataset to filter
        """
        self.dataset = dataset
        self.original_size = len(dataset)

    def apply(self) -> Dataset:
        """
        Apply the filter to the dataset.

        Returns:
            Filtered dataset

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def get_filter_stats(self) -> Dict[str, int]:
        """
        Get statistics about the filtering operation.

        Returns:
            Dictionary with filter statistics
        """
        return {
            "original_size": self.original_size,
            "filtered_size": len(self.dataset),
            "removed_count": self.original_size - len(self.dataset),
        }


class AmbiguousQuestionFilter(DatasetFilter):
    """
    Filter dataset based on cartography metrics to remove ambiguous examples.

    Keeps:
      - All non-ambiguous examples (easy + hard)
      - Only the top `top_fraction` most ambiguous examples
        (based on instability = (1 - confidence) + variability)
    """

    def __init__(
        self,
        dataset: Dataset,
        metrics_path: str,
        top_fraction: float = 0.33,
        variability_margin: float = 0.0,
        apply_rule_based_filter: bool = False,
    ):
        """
        Initialize the ambiguous question filter.

        Args:
            dataset: Dataset to filter
            metrics_path: Path to cartography output directory or CSV file
            top_fraction: Fraction of most ambiguous examples to keep (default: 0.33 = top 33%)
            variability_margin: Margin to adjust variability threshold (default: 0.0). Should be set to train_variability_margin or val_variability_margin depending on split.
            apply_rule_based_filter: If True, also filter non-questions from top ambiguous examples
        Note:
            Pass the correct margin for the split (train or validation) via config as train_variability_margin or val_variability_margin.
        """
        super().__init__(dataset)
        self.metrics_path = metrics_path
        self.top_fraction = top_fraction
        self.variability_margin = variability_margin  # Should be set per split (train/val)
        self.apply_rule_based_filter = apply_rule_based_filter
        self.stats = {
            "removed_ambiguous_not_top": 0,
            "removed_nonquestion_top_ambiguous": 0,
        }

    def apply(self) -> Dataset:
        """
        Apply the ambiguous question filter.

        Returns:
            Filtered dataset
        """
        try:
            # Load cartography metrics
            cartography_df = self._load_metrics()

            # Categorize examples
            cartography_df = categorize_examples(
                cartography_df, variability_margin=self.variability_margin
            )

            # Get ambiguous examples
            ambiguous_df = cartography_df[
                cartography_df["category"] == "ambiguous"
            ].copy()

            if ambiguous_df.empty:
                print(
                    "No ambiguous examples found in cartography metrics; returning original dataset."
                )
                return self.dataset

            # Compute instability score
            ambiguous_df["instability"] = (
                1.0 - ambiguous_df["confidence"]
            ) + ambiguous_df["variability"]

            # Find threshold for top `top_fraction` most ambiguous
            keep_quantile = 1.0 - self.top_fraction
            threshold = ambiguous_df["instability"].quantile(keep_quantile)

            # Get IDs of top ambiguous examples
            top_ambiguous_df = ambiguous_df[ambiguous_df["instability"] >= threshold]
            top_ambiguous_ids = set(top_ambiguous_df.index.tolist())

            # All ambiguous IDs
            all_ambiguous_ids = set(ambiguous_df.index.tolist())

            # Filter dataset
            filtered_dataset = self._filter_examples(
                all_ambiguous_ids, top_ambiguous_ids
            )

            # Print statistics
            self._print_stats(filtered_dataset)

            return filtered_dataset

        except Exception as e:
            print(f"Error during ambiguous filtering: {e}")
            print("Returning original dataset without filtering...")
            return self.dataset

    def _load_metrics(self) -> pd.DataFrame:
        """Load cartography metrics from file or directory."""
        # Check if it's a directory or file
        if os.path.isdir(self.metrics_path):
            return load_cartography_metrics(self.metrics_path)
        elif os.path.isfile(self.metrics_path):
            # Load CSV directly
            return pd.read_csv(self.metrics_path, index_col="id")
        else:
            raise FileNotFoundError(
                f"Cartography metrics not found: {self.metrics_path}"
            )

    def _filter_examples(
        self, all_ambiguous_ids: set, top_ambiguous_ids: set
    ) -> Dataset:
        """Filter examples based on ambiguity and optional rule-based criteria."""
        df = self.dataset.to_pandas()
        keep_indices = []

        for idx, row in df.iterrows():
            example_id = row.get("id")
            question = row.get("question", "")

            # Check if example is ambiguous
            if example_id in all_ambiguous_ids:
                # Only keep if in top fraction of most ambiguous
                if example_id not in top_ambiguous_ids:
                    self.stats["removed_ambiguous_not_top"] += 1
                    continue

                # Optionally apply rule-based filter for non-questions
                if self.apply_rule_based_filter:
                    from rule_based_errors import rule9_question_not_starting_with_qword

                    if rule9_question_not_starting_with_qword(question):
                        self.stats["removed_nonquestion_top_ambiguous"] += 1
                        continue

                keep_indices.append(idx)
            else:
                # Non-ambiguous examples are always kept
                keep_indices.append(idx)

        filtered_df = df.iloc[keep_indices].reset_index(drop=True)
        return Dataset.from_pandas(filtered_df)

    def _print_stats(self, filtered_dataset: Dataset):
        """Print filtering statistics."""
        print(f"Original size: {self.original_size}")
        print(f"Filtered size: {len(filtered_dataset)}")
        print(
            f"Dropped ambiguous not in top {int(self.top_fraction * 100)}%: {self.stats['removed_ambiguous_not_top']}"
        )
        if self.apply_rule_based_filter:
            print(
                f"Dropped non-questions from top ambiguous: {self.stats['removed_nonquestion_top_ambiguous']}"
            )


# Currenlty not used.
class CategoryFilter(DatasetFilter):
    """
    Filter dataset to keep only examples from specific cartography categories.

    Categories: 'easy', 'hard', 'ambiguous'
    """

    def __init__(
        self,
        dataset: Dataset,
        metrics_path: str,
        categories: List[str],
    ):
        """
        Initialize the category filter.

        Args:
            dataset: Dataset to filter
            metrics_path: Path to cartography output directory or CSV file
            categories: List of categories to keep (e.g., ['easy', 'hard'])
        """
        super().__init__(dataset)
        self.metrics_path = metrics_path
        self.categories = categories
        self.stats = {}

    def apply(self) -> Dataset:
        """
        Apply the category filter.

        Returns:
            Filtered dataset with only specified categories
        """
        try:
            # Load and categorize
            cartography_df = self._load_metrics()
            cartography_df = categorize_examples(cartography_df)

            # Get IDs of examples in specified categories
            keep_ids = set(
                cartography_df[cartography_df["category"].isin(self.categories)].index
            )

            # Filter dataset
            df = self.dataset.to_pandas()
            keep_indices = [
                idx for idx, row in df.iterrows() if row.get("id") in keep_ids
            ]

            filtered_df = df.iloc[keep_indices].reset_index(drop=True)
            filtered_dataset = Dataset.from_pandas(filtered_df)

            # Track stats
            for cat in self.categories:
                cat_count = (cartography_df["category"] == cat).sum()
                self.stats[cat] = cat_count

            self._print_stats(filtered_dataset)

            return filtered_dataset

        except Exception as e:
            print(f"Error during category filtering: {e}")
            print("Returning original dataset without filtering...")
            return self.dataset

    def _load_metrics(self) -> pd.DataFrame:
        """Load cartography metrics from file or directory."""
        if os.path.isdir(self.metrics_path):
            return load_cartography_metrics(self.metrics_path)
        elif os.path.isfile(self.metrics_path):
            return pd.read_csv(self.metrics_path, index_col="example_id")
        else:
            raise FileNotFoundError(
                f"Cartography metrics not found: {self.metrics_path}"
            )

    def _print_stats(self, filtered_dataset: Dataset):
        """Print filtering statistics."""
        print(f"Original size: {self.original_size}")
        print(f"Filtered size: {len(filtered_dataset)}")
        print(f"Kept categories: {', '.join(self.categories)}")
        for cat, count in self.stats.items():
            print(f"  - {cat}: {count} examples")


class ClusterFilter(DatasetFilter):
    """
    Filter dataset based on cluster assignments.

    Exclude examples from specific clusters (e.g., noise or unwanted semantic groups).
    """

    def __init__(
        self,
        dataset: Dataset,
        cluster_path: str,
        exclude_clusters: List[int] = None,
        min_probability: Optional[float] = None,
    ):
        """
        Initialize the cluster filter.

        Args:
            dataset: Dataset to filter
            cluster_path: Path to cluster output directory or CSV file
            exclude_clusters: List of cluster IDs to exclude (e.g., [3, 4, -1]).
                            Default: [-1] (excludes noise). Use [] to keep all clusters.
            min_probability: Minimum cluster probability threshold (if available)
        """
        super().__init__(dataset)
        self.cluster_path = cluster_path
        self.exclude_clusters = (
            exclude_clusters if exclude_clusters is not None else [-1]
        )
        self.min_probability = min_probability
        self.stats = {}

    def apply(self) -> Dataset:
        """
        Apply the cluster filter.

        Returns:
            Filtered dataset excluding specified clusters
        """
        try:
            # Load cluster assignments
            cluster_df = load_cluster_assignments(self.cluster_path)

            # Start with all IDs
            keep_ids = set(cluster_df.index)

            # Exclude specified clusters
            if self.exclude_clusters:
                exclude_ids = set(
                    cluster_df[cluster_df["cluster"].isin(self.exclude_clusters)].index
                )
                keep_ids = keep_ids - exclude_ids

            # Apply probability threshold if specified
            if (
                self.min_probability is not None
                and "cluster_probability" in cluster_df.columns
            ):
                low_prob_ids = set(
                    cluster_df[
                        cluster_df["cluster_probability"] < self.min_probability
                    ].index
                )
                keep_ids = keep_ids - low_prob_ids
                self.stats["removed_low_probability"] = len(low_prob_ids)

            # Filter dataset
            df = self.dataset.to_pandas()
            keep_indices = [
                idx for idx, row in df.iterrows() if row.get("id") in keep_ids
            ]

            filtered_df = df.iloc[keep_indices].reset_index(drop=True)
            filtered_dataset = Dataset.from_pandas(filtered_df)

            # Track cluster statistics
            kept_cluster_df = cluster_df[cluster_df.index.isin(keep_ids)]
            for cluster_id in kept_cluster_df["cluster"].unique():
                count = (kept_cluster_df["cluster"] == cluster_id).sum()
                cluster_label = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
                self.stats[cluster_label] = count

            self._print_stats(filtered_dataset)

            return filtered_dataset

        except Exception as e:
            print(f"Error during cluster filtering: {e}")
            print("Returning original dataset without filtering...")
            return self.dataset

    def _print_stats(self, filtered_dataset: Dataset):
        """Print filtering statistics."""
        print(f"Original size: {self.original_size}")
        print(f"Filtered size: {len(filtered_dataset)}")
        print("Clusters kept:")
        for cluster_label, count in sorted(self.stats.items()):
            if not cluster_label.startswith("removed"):
                print(f"  - {cluster_label}: {count} examples")
        if "removed_low_probability" in self.stats:
            print(
                f"Removed low probability: {self.stats['removed_low_probability']} examples"
            )


class RuleBasedFilter(DatasetFilter):
    """
    Filter dataset based on rule-based error detection.

    Removes examples that match the specified rule (i.e., rule returns True).
    Supports any rule from rule_based_errors module.
    """

    def __init__(
        self,
        dataset: Dataset,
        rule_name: str = "low_answer_question_overlap",
        sim_threshold: float = 0.05,
    ):
        """
        Initialize the rule-based filter.

        Args:
            dataset: Dataset to filter
            rule_name: Name of the rule to apply (default: "low_answer_question_overlap")
            sim_threshold: Similarity threshold for overlap rules (default: 0.05)
        """
        super().__init__(dataset)
        self.rule_name = rule_name
        self.sim_threshold = sim_threshold
        self.stats = {"removed_by_rule": 0}

    def apply(self) -> Dataset:
        """
        Apply the rule-based filter.

        Returns:
            Filtered dataset
        """
        try:
            # Import the rule function
            from rule_based_errors import low_answer_question_overlap

            # Map rule names to functions
            rule_functions = {
                "low_answer_question_overlap": low_answer_question_overlap,
            }

            if self.rule_name not in rule_functions:
                print(
                    f"Warning: Unknown rule '{self.rule_name}'. Returning original dataset."
                )
                return self.dataset

            rule_func = rule_functions[self.rule_name]

            df = self.dataset.to_pandas()
            keep_indices = []

            for idx, row in df.iterrows():
                # Extract answer text from the answers field
                answer = row.get("answers", {})
                if isinstance(answer, dict):
                    answer_texts = answer.get("text", [])
                    answer_text = answer_texts[0] if answer_texts else ""
                else:
                    answer_text = str(answer) if answer else ""

                question = row.get("question", "")

                # Apply rule - if it returns True, we EXCLUDE the example
                try:
                    should_filter = rule_func(answer_text, question, self.sim_threshold)
                    if should_filter:
                        self.stats["removed_by_rule"] += 1
                        continue
                except Exception as e:
                    # If rule fails, keep the example to be safe
                    print(f"Warning: Rule failed for example {row.get('id', idx)}: {e}")

                keep_indices.append(idx)

            filtered_df = df.iloc[keep_indices].reset_index(drop=True)
            filtered_dataset = Dataset.from_pandas(filtered_df)

            self._print_stats(filtered_dataset)

            return filtered_dataset

        except Exception as e:
            print(f"Error during rule-based filtering: {e}")
            import traceback

            traceback.print_exc()
            print("Returning original dataset without filtering...")
            return self.dataset

    def _print_stats(self, filtered_dataset: Dataset):
        """Print filtering statistics."""
        print(f"Original size: {self.original_size}")
        print(f"Filtered size: {len(filtered_dataset)}")
        print(f"Removed by {self.rule_name}: {self.stats['removed_by_rule']}")
        print(f"Similarity threshold: {self.sim_threshold}")


# Currenlty not used.
class ConfidenceThresholdFilter(DatasetFilter):
    """
    Filter dataset based on confidence thresholds.

    Keep only examples with confidence within specified range.
    """

    def __init__(
        self,
        dataset: Dataset,
        metrics_path: str,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
    ):
        """
        Initialize the confidence threshold filter.

        Args:
            dataset: Dataset to filter
            metrics_path: Path to cartography output directory or CSV file
            min_confidence: Minimum confidence threshold (inclusive)
            max_confidence: Maximum confidence threshold (inclusive)
        """
        super().__init__(dataset)
        self.metrics_path = metrics_path
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def apply(self) -> Dataset:
        """
        Apply the confidence threshold filter.

        Returns:
            Filtered dataset
        """
        try:
            # Load metrics
            cartography_df = self._load_metrics()

            # Apply thresholds
            mask = pd.Series([True] * len(cartography_df), index=cartography_df.index)

            if self.min_confidence is not None:
                mask &= cartography_df["confidence"] >= self.min_confidence

            if self.max_confidence is not None:
                mask &= cartography_df["confidence"] <= self.max_confidence

            keep_ids = set(cartography_df[mask].index)

            # Filter dataset
            df = self.dataset.to_pandas()
            keep_indices = [
                idx for idx, row in df.iterrows() if row.get("id") in keep_ids
            ]

            filtered_df = df.iloc[keep_indices].reset_index(drop=True)
            filtered_dataset = Dataset.from_pandas(filtered_df)

            self._print_stats(filtered_dataset)

            return filtered_dataset

        except Exception as e:
            print(f"Error during confidence threshold filtering: {e}")
            print("Returning original dataset without filtering...")
            return self.dataset

    def _load_metrics(self) -> pd.DataFrame:
        """Load cartography metrics from file or directory."""
        if os.path.isdir(self.metrics_path):
            return load_cartography_metrics(self.metrics_path)
        elif os.path.isfile(self.metrics_path):
            return pd.read_csv(self.metrics_path, index_col="example_id")
        else:
            raise FileNotFoundError(
                f"Cartography metrics not found: {self.metrics_path}"
            )

    def _print_stats(self, filtered_dataset: Dataset):
        """Print filtering statistics."""
        print(f"Original size: {self.original_size}")
        print(f"Filtered size: {len(filtered_dataset)}")
        if self.min_confidence is not None:
            print(f"Min confidence: {self.min_confidence}")
        if self.max_confidence is not None:
            print(f"Max confidence: {self.max_confidence}")


def apply_filters(
    dataset: Dataset,
    filter_config: Dict,
) -> Dataset:
    """
    Apply multiple filters to a dataset based on configuration.

    Args:
        dataset: Dataset to filter
        filter_config: Dictionary with filter configuration. Example:
            {
                "ambiguous": {
                    "enabled": True,
                    "metrics_path": "./cartography_output",
                    "top_fraction": 0.33,
                    "apply_rule_based_filter": False
                },
                "category": {
                    "enabled": False,
                    "metrics_path": "./cartography_output",
                    "categories": ["easy", "hard"]
                },
                "confidence": {
                    "enabled": False,
                    "metrics_path": "./cartography_output",
                    "min_confidence": 0.5,
                    "max_confidence": None
                },
                "cluster": {
                    "enabled": False,
                    "cluster_path": "./cluster_output",
                    "exclude_clusters": [-1],
                    "min_probability": None
                }
            }

    Returns:
        Filtered dataset
    """
    filtered_dataset = dataset

    # Apply ambiguous question filter
    if filter_config.get("ambiguous", {}).get("enabled", False):
        config = filter_config["ambiguous"]
        print("\n" + "=" * 70)
        print("Applying Ambiguous Question Filter")
        print("=" * 70)
        # Pass the correct margin for the split (train/val) as variability_margin
        filter_obj = AmbiguousQuestionFilter(
            dataset=filtered_dataset,
            metrics_path=config["metrics_path"],
            top_fraction=config.get("top_fraction", 0.33),
            variability_margin=config.get("variability_margin", 0.0),  # Should be set to train_variability_margin or val_variability_margin in config
            apply_rule_based_filter=config.get("apply_rule_based_filter", False),
        )
        filtered_dataset = filter_obj.apply()
        print("=" * 70 + "\n")

    # Apply category filter
    if filter_config.get("category", {}).get("enabled", False):
        config = filter_config["category"]
        print("\n" + "=" * 70)
        print("Applying Category Filter")
        print("=" * 70)
        filter_obj = CategoryFilter(
            dataset=filtered_dataset,
            metrics_path=config["metrics_path"],
            categories=config["categories"],
        )
        filtered_dataset = filter_obj.apply()
        print("=" * 70 + "\n")

    # Apply confidence threshold filter
    if filter_config.get("confidence", {}).get("enabled", False):
        config = filter_config["confidence"]
        print("\n" + "=" * 70)
        print("Applying Confidence Threshold Filter")
        print("=" * 70)
        filter_obj = ConfidenceThresholdFilter(
            dataset=filtered_dataset,
            metrics_path=config["metrics_path"],
            min_confidence=config.get("min_confidence"),
            max_confidence=config.get("max_confidence"),
        )
        filtered_dataset = filter_obj.apply()
        print("=" * 70 + "\n")

    # Apply cluster filter
    if filter_config.get("cluster", {}).get("enabled", False):
        config = filter_config["cluster"]
        print("\n" + "=" * 70)
        print("Applying Cluster Filter")
        print("=" * 70)
        filter_obj = ClusterFilter(
            dataset=filtered_dataset,
            cluster_path=config["cluster_path"],
            exclude_clusters=config.get("exclude_clusters", [-1]),
            min_probability=config.get("min_probability"),
        )
        filtered_dataset = filter_obj.apply()
        print("=" * 70 + "\n")

    # Apply rule-based filter
    if filter_config.get("rule_based", {}).get("enabled", False):
        config = filter_config["rule_based"]
        print("\n" + "=" * 70)
        print("Applying Rule-Based Filter")
        print("=" * 70)
        filter_obj = RuleBasedFilter(
            dataset=filtered_dataset,
            rule_name=config.get("rule_name", "low_answer_question_overlap"),
            sim_threshold=config.get("sim_threshold", 0.05),
        )
        filtered_dataset = filter_obj.apply()
        print("=" * 70 + "\n")

    return filtered_dataset

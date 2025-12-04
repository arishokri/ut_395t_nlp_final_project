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

from analyze_cartography import categorize_examples, load_cartography_metrics


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
        apply_rule_based_filter: bool = False,
    ):
        """
        Initialize the ambiguous question filter.

        Args:
            dataset: Dataset to filter
            metrics_path: Path to cartography output directory or CSV file
            top_fraction: Fraction of most ambiguous examples to keep (default: 0.33 = top 33%)
            apply_rule_based_filter: If True, also filter non-questions from top ambiguous examples
        """
        super().__init__(dataset)
        self.metrics_path = metrics_path
        self.top_fraction = top_fraction
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
            cartography_df = categorize_examples(cartography_df)

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
            return pd.read_csv(self.metrics_path, index_col="example_id")
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
        filter_obj = AmbiguousQuestionFilter(
            dataset=filtered_dataset,
            metrics_path=config["metrics_path"],
            top_fraction=config.get("top_fraction", 0.33),
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

    return filtered_dataset

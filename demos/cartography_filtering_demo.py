"""
Example script demonstrating dataset filtering usage.

This script shows various ways to filter datasets based on cartography metrics.
"""

# Making sure this can be run from outside the directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import datasets

from dataset_filters import (
    AmbiguousQuestionFilter,
    CategoryFilter,
    ConfidenceThresholdFilter,
    apply_filters,
)
from helpers import generate_hash_ids


def example_1_basic_ambiguous_filter():
    """Example 1: Basic ambiguous question filtering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Ambiguous Question Filter")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_dataset = dataset["train"]

    # Add IDs if missing
    if "id" not in train_dataset.column_names:
        train_dataset = train_dataset.map(generate_hash_ids)

    # Limit to first 1000 examples for demo
    train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

    print(f"Original dataset size: {len(train_dataset)}")

    # Apply filter
    filter_obj = AmbiguousQuestionFilter(
        dataset=train_dataset,
        metrics_path="./cartography_output",
        top_fraction=0.33,  # Keep top 33% most ambiguous
    )
    filtered_dataset = filter_obj.apply()

    print(f"Filtered dataset size: {len(filtered_dataset)}")
    return filtered_dataset


def example_2_category_filter():
    """Example 2: Filter by category (keep only easy + hard examples)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Category Filter (Easy + Hard only)")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_dataset = dataset["train"]

    # Add IDs if missing
    if "id" not in train_dataset.column_names:
        train_dataset = train_dataset.map(generate_hash_ids)

    # Limit to first 1000 examples for demo
    train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

    print(f"Original dataset size: {len(train_dataset)}")

    # Apply filter - keep only easy and hard, drop all ambiguous
    filter_obj = CategoryFilter(
        dataset=train_dataset,
        metrics_path="./cartography_output",
        categories=["easy", "hard"],
    )
    filtered_dataset = filter_obj.apply()

    print(f"Filtered dataset size: {len(filtered_dataset)}")
    return filtered_dataset


def example_3_confidence_threshold():
    """Example 3: Filter by confidence threshold."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Confidence Threshold Filter")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_dataset = dataset["train"]

    # Add IDs if missing
    if "id" not in train_dataset.column_names:
        train_dataset = train_dataset.map(generate_hash_ids)

    # Limit to first 1000 examples for demo
    train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

    print(f"Original dataset size: {len(train_dataset)}")

    # Keep only examples with moderate confidence (0.3 to 0.8)
    filter_obj = ConfidenceThresholdFilter(
        dataset=train_dataset,
        metrics_path="./cartography_output",
        min_confidence=0.3,
        max_confidence=0.8,
    )
    filtered_dataset = filter_obj.apply()

    print(f"Filtered dataset size: {len(filtered_dataset)}")
    return filtered_dataset


def example_4_multiple_filters():
    """Example 4: Apply multiple filters using apply_filters()."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Multiple Filters (Ambiguous + Confidence)")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_dataset = dataset["train"]

    # Add IDs if missing
    if "id" not in train_dataset.column_names:
        train_dataset = train_dataset.map(generate_hash_ids)

    # Limit to first 1000 examples for demo
    train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

    print(f"Original dataset size: {len(train_dataset)}")

    # Define filter configuration
    filter_config = {
        "ambiguous": {
            "enabled": True,
            "metrics_path": "./cartography_output",
            "top_fraction": 0.25,  # Keep top 25% most ambiguous
        },
        "confidence": {
            "enabled": True,
            "metrics_path": "./cartography_output",
            "min_confidence": 0.2,
            "max_confidence": None,  # No upper limit
        },
    }

    # Apply all filters
    filtered_dataset = apply_filters(train_dataset, filter_config)

    print(f"Final filtered dataset size: {len(filtered_dataset)}")
    return filtered_dataset


def example_5_only_easy():
    """Example 5: Keep only easy examples for curriculum learning."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Curriculum Learning (Easy Examples Only)")
    print("=" * 70)

    # Load dataset
    dataset = datasets.load_dataset("Eladio/emrqa-msquad")
    train_dataset = dataset["train"]

    # Add IDs if missing
    if "id" not in train_dataset.column_names:
        train_dataset = train_dataset.map(generate_hash_ids)

    # Limit to first 1000 examples for demo
    train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

    print(f"Original dataset size: {len(train_dataset)}")

    # Keep only easy examples
    filter_obj = CategoryFilter(
        dataset=train_dataset,
        metrics_path="./cartography_output",
        categories=["easy"],
    )
    filtered_dataset = filter_obj.apply()

    print(f"Filtered dataset size: {len(filtered_dataset)}")
    return filtered_dataset


def main():
    """Run all examples."""
    print("\n" + "#" * 70)
    print("# Dataset Filtering Examples")
    print("#" * 70)
    print("\nNote: These examples require cartography metrics to be pre-computed.")
    print("Run training with --enable_cartography first if metrics don't exist.\n")

    try:
        # Run each example
        example_1_basic_ambiguous_filter()
        example_2_category_filter()
        example_3_confidence_threshold()
        example_4_multiple_filters()
        example_5_only_easy()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print("\n" + "!" * 70)
        print("ERROR: Cartography metrics not found!")
        print("!" * 70)
        print(f"\n{e}\n")
        print("To generate cartography metrics, run:")
        print(
            "  python run.py --do_train --enable_cartography --num_train_epochs 3 --output_dir ./example_model/"
        )
        print("\nThen re-run this script.\n")


if __name__ == "__main__":
    main()

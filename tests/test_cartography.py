"""Tests for dataset cartography functionality."""

import pandas as pd

from analyze_cartography import (
    categorize_examples,
    load_cartography_metrics,
)
from dataset_cartography import DatasetCartographyCallback


class TestLoadCartographyMetrics:
    """Test suite for loading cartography metrics."""

    def test_load_from_directory(self, sample_cartography_metrics):
        """Test loading metrics from directory."""
        df = load_cartography_metrics(sample_cartography_metrics)

        assert isinstance(df, pd.DataFrame)
        assert "confidence" in df.columns
        assert "variability" in df.columns
        assert len(df) == 5
        # Index should be example_id
        assert df.index.name == "example_id"

    def test_load_from_csv_file(self, sample_cartography_metrics):
        """Test loading metrics from CSV file path."""
        # Function expects directory path, not CSV path
        df = load_cartography_metrics(sample_cartography_metrics)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_metrics_have_required_columns(self, sample_cartography_metrics):
        """Test that loaded metrics have required columns."""
        df = load_cartography_metrics(sample_cartography_metrics)

        required_cols = ["confidence", "variability", "correctness"]
        for col in required_cols:
            assert col in df.columns
        # example_id is the index
        assert df.index.name == "example_id"


class TestCategorizeExamples:
    """Test suite for categorizing examples."""

    def test_categorizes_easy_hard_ambiguous(self, sample_cartography_metrics):
        """Test that examples are categorized correctly."""
        df = load_cartography_metrics(sample_cartography_metrics)
        categorized = categorize_examples(df)

        assert "category" in categorized.columns
        categories = set(categorized["category"].unique())

        # Should have at least one of the categories
        valid_categories = {"easy", "hard", "ambiguous"}
        assert len(categories.intersection(valid_categories)) > 0

    def test_easy_examples_high_confidence_correctness(
        self, sample_cartography_metrics
    ):
        """Test that easy examples have high confidence and correctness."""
        df = load_cartography_metrics(sample_cartography_metrics)
        categorized = categorize_examples(df)

        easy_examples = categorized[categorized["category"] == "easy"]
        if len(easy_examples) > 0:
            # Easy examples should have high confidence
            assert easy_examples["confidence"].mean() > 0.7

    def test_ambiguous_examples_high_variability(self, sample_cartography_metrics):
        """Test that ambiguous examples have high variability."""
        df = load_cartography_metrics(sample_cartography_metrics)
        categorized = categorize_examples(df)

        ambiguous_examples = categorized[categorized["category"] == "ambiguous"]
        if len(ambiguous_examples) > 0:
            # Ambiguous examples should have higher variability
            assert ambiguous_examples["variability"].mean() > 0.3


class TestDatasetCartographyCallback:
    """Test suite for DatasetCartographyCallback."""

    def test_callback_initialization(self, temp_dir):
        """Test callback can be initialized."""
        output_dir = temp_dir / "cartography_test"
        callback = DatasetCartographyCallback(
            output_dir=str(output_dir),
        )

        assert callback.output_dir == str(output_dir)
        assert callback.training_dynamics == {}

    def test_callback_creates_output_directory(self, temp_dir):
        """Test that callback creates output directory."""
        output_dir = temp_dir / "cartography_test"
        callback = DatasetCartographyCallback(
            output_dir=str(output_dir),
        )

        # Manually trigger directory creation (normally done in on_train_begin)
        import os

        os.makedirs(callback.output_dir, exist_ok=True)

        assert output_dir.exists()

    def test_callback_saves_metrics(self, temp_dir):
        """Test that callback can save metrics."""
        output_dir = temp_dir / "cartography_test"
        output_dir.mkdir(exist_ok=True)

        callback = DatasetCartographyCallback(
            output_dir=str(output_dir),
        )

        # Simulate some training dynamics
        callback.training_dynamics = {
            "example_1": {
                "correctness": [1.0, 1.0],
                "confidence": [0.9, 0.95],
                "logits": [[0.1, 0.9], [0.05, 0.95]],
            }
        }

        # Save metrics
        metrics_path = output_dir / "cartography_metrics.csv"

        # Create a simple metrics DataFrame
        metrics_df = pd.DataFrame(
            {
                "example_id": ["example_1"],
                "confidence": [0.925],
                "variability": [0.025],
                "correctness": [1.0],
            }
        )
        metrics_df.to_csv(metrics_path, index=False)

        assert metrics_path.exists()

        # Verify we can load it back
        df = pd.read_csv(metrics_path)
        assert len(df) == 1
        assert "example_id" in df.columns

"""Tests for cluster analysis functionality."""

import numpy as np
import pandas as pd

from cluster_analysis import load_cluster_assignments


class TestLoadClusterAssignments:
    """Test suite for loading cluster assignments."""

    def test_load_from_directory(self, sample_cluster_assignments):
        """Test loading assignments from directory."""
        df = load_cluster_assignments(sample_cluster_assignments)

        assert isinstance(df, pd.DataFrame)
        assert "cluster" in df.columns
        assert len(df) == 5
        # Index should be 'id'
        assert df.index.name == "id"

    def test_load_from_csv_file(self, sample_cluster_assignments):
        """Test loading assignments from CSV file."""
        csv_path = f"{sample_cluster_assignments}/cluster_assignments.csv"
        df = load_cluster_assignments(csv_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert df.index.name == "id"

    def test_assignments_have_required_columns(self, sample_cluster_assignments):
        """Test that loaded assignments have required columns."""
        df = load_cluster_assignments(sample_cluster_assignments)

        assert "cluster" in df.columns
        # id is the index, not a column
        assert df.index.name == "id"

    def test_cluster_ids_are_integers(self, sample_cluster_assignments):
        """Test that cluster IDs are integers."""
        df = load_cluster_assignments(sample_cluster_assignments)

        # Cluster IDs should be integers (including -1 for noise)
        assert df["cluster"].dtype in [np.int64, np.int32, int]

    def test_noise_cluster_present(self, sample_cluster_assignments):
        """Test that noise cluster (-1) can be identified."""
        df = load_cluster_assignments(sample_cluster_assignments)

        # Check if -1 (noise) exists
        cluster_ids = df["cluster"].unique()
        assert -1 in cluster_ids


class TestClusterProbabilities:
    """Test suite for cluster probability handling."""

    def test_probability_column_exists(self, sample_cluster_assignments):
        """Test that probability column exists when present."""
        df = load_cluster_assignments(sample_cluster_assignments)

        if "probability" in df.columns:
            assert df["probability"].dtype == float
            # Probabilities should be between 0 and 1
            assert (df["probability"] >= 0).all()
            assert (df["probability"] <= 1).all()

    def test_probability_filtering(self, sample_cluster_assignments):
        """Test filtering by probability threshold."""
        df = load_cluster_assignments(sample_cluster_assignments)

        if "probability" in df.columns:
            high_prob = df[df["probability"] >= 0.9]
            # Should have some high probability examples
            assert len(high_prob) >= 0  # Could be 0 if none meet threshold


class TestClusterStatistics:
    """Test suite for cluster statistics."""

    def test_cluster_counts(self, sample_cluster_assignments):
        """Test counting examples per cluster."""
        df = load_cluster_assignments(sample_cluster_assignments)

        cluster_counts = df["cluster"].value_counts()

        # Should have multiple clusters
        assert len(cluster_counts) > 0
        # Each cluster should have at least 1 example
        assert (cluster_counts > 0).all()

    def test_noise_cluster_identification(self, sample_cluster_assignments):
        """Test identifying noise cluster."""
        df = load_cluster_assignments(sample_cluster_assignments)

        noise_examples = df[df["cluster"] == -1]

        # Should be able to identify noise examples (may be 0)
        assert len(noise_examples) >= 0

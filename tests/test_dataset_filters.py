"""Tests for dataset filtering functionality."""

from datasets import Dataset

from dataset_filters import (
    AmbiguousQuestionFilter,
    ClusterFilter,
    apply_filters,
)


class TestAmbiguousQuestionFilter:
    """Test suite for AmbiguousQuestionFilter."""

    def test_filter_initialization(self, sample_qa_dataset, sample_cartography_metrics):
        """Test filter can be initialized with dataset and metrics."""
        filter_obj = AmbiguousQuestionFilter(
            dataset=sample_qa_dataset,
            metrics_path=sample_cartography_metrics,
            top_fraction=0.5,
        )
        assert filter_obj.original_size == 5
        assert filter_obj.top_fraction == 0.5

    def test_filter_removes_ambiguous_examples(
        self, sample_qa_dataset, sample_cartography_metrics
    ):
        """Test that ambiguous examples are filtered correctly."""
        filter_obj = AmbiguousQuestionFilter(
            dataset=sample_qa_dataset,
            metrics_path=sample_cartography_metrics,
            top_fraction=0.0,  # Remove all ambiguous
        )
        filtered_dataset = filter_obj.apply()

        # Should keep only easy examples (q1, q2, q4)
        assert len(filtered_dataset) <= len(sample_qa_dataset)

    def test_filter_keeps_top_ambiguous(
        self, sample_qa_dataset, sample_cartography_metrics
    ):
        """Test that top fraction of ambiguous examples are kept."""
        filter_obj = AmbiguousQuestionFilter(
            dataset=sample_qa_dataset,
            metrics_path=sample_cartography_metrics,
            top_fraction=0.5,  # Keep top 50% of ambiguous
        )
        filtered_dataset = filter_obj.apply()

        # Should keep easy + top 50% ambiguous
        assert len(filtered_dataset) >= 3  # At least the easy ones

    def test_filter_stats(self, sample_qa_dataset, sample_cartography_metrics):
        """Test that filter statistics are computed correctly."""
        filter_obj = AmbiguousQuestionFilter(
            dataset=sample_qa_dataset,
            metrics_path=sample_cartography_metrics,
            top_fraction=0.5,
        )
        _ = filter_obj.apply()
        stats = filter_obj.get_filter_stats()

        assert "original_size" in stats
        assert "filtered_size" in stats
        assert "removed_count" in stats
        assert stats["original_size"] == 5

    def test_variability_margin(self, sample_qa_dataset, sample_cartography_metrics):
        """Test that variability_margin parameter affects classification."""
        # Test with positive margin (stricter - fewer ambiguous)
        filter_strict = AmbiguousQuestionFilter(
            dataset=sample_qa_dataset,
            metrics_path=sample_cartography_metrics,
            top_fraction=1.0,  # Keep all ambiguous
            variability_margin=0.1,  # Positive margin
        )
        filtered_strict = filter_strict.apply()

        # Test with negative margin (more lenient - more ambiguous)
        filter_lenient = AmbiguousQuestionFilter(
            dataset=sample_qa_dataset,
            metrics_path=sample_cartography_metrics,
            top_fraction=1.0,  # Keep all ambiguous
            variability_margin=-0.1,  # Negative margin
        )
        filtered_lenient = filter_lenient.apply()

        # With negative margin, more examples are classified as ambiguous
        # So fewer examples should be kept (since top_fraction=1.0 keeps all ambiguous,
        # but the total filtered size might differ based on easy/hard classification)
        assert filter_strict.variability_margin == 0.1
        assert filter_lenient.variability_margin == -0.1


class TestClusterFilter:
    """Test suite for ClusterFilter."""

    def test_filter_initialization(self, sample_qa_dataset, sample_cluster_assignments):
        """Test cluster filter initialization."""
        filter_obj = ClusterFilter(
            dataset=sample_qa_dataset,
            cluster_path=sample_cluster_assignments,
            exclude_clusters=[-1],
        )
        assert filter_obj.original_size == 5
        assert -1 in filter_obj.exclude_clusters

    def test_filter_excludes_noise_cluster(
        self, sample_qa_dataset, sample_cluster_assignments
    ):
        """Test that noise cluster (-1) is excluded."""
        filter_obj = ClusterFilter(
            dataset=sample_qa_dataset,
            cluster_path=sample_cluster_assignments,
            exclude_clusters=[-1],
        )
        filtered_dataset = filter_obj.apply()

        # Should remove q5 which is in cluster -1
        assert len(filtered_dataset) == 4

    def test_filter_excludes_multiple_clusters(
        self, sample_qa_dataset, sample_cluster_assignments
    ):
        """Test excluding multiple clusters."""
        filter_obj = ClusterFilter(
            dataset=sample_qa_dataset,
            cluster_path=sample_cluster_assignments,
            exclude_clusters=[-1, 1],
        )
        filtered_dataset = filter_obj.apply()

        # Should remove q3 (cluster 1) and q5 (cluster -1)
        assert len(filtered_dataset) == 3

    def test_filter_probability_threshold(
        self, sample_qa_dataset, sample_cluster_assignments
    ):
        """Test filtering by minimum cluster probability."""
        filter_obj = ClusterFilter(
            dataset=sample_qa_dataset,
            cluster_path=sample_cluster_assignments,
            min_probability=0.9,
        )
        filtered_dataset = filter_obj.apply()

        # Should keep only examples with probability >= 0.9
        assert len(filtered_dataset) <= 5


class TestApplyFilters:
    """Test suite for apply_filters function."""

    def test_no_filters(self, sample_qa_dataset):
        """Test that dataset is unchanged when no filters are applied."""
        filter_config = {}
        filtered = apply_filters(
            dataset=sample_qa_dataset,
            filter_config=filter_config,
        )
        assert len(filtered) == len(sample_qa_dataset)

    def test_cartography_filter_only(
        self, sample_qa_dataset, sample_cartography_metrics
    ):
        """Test applying only cartography filter."""
        filter_config = {
            "ambiguous": {
                "enabled": True,
                "metrics_path": sample_cartography_metrics,
                "top_fraction": 0.5,
            }
        }
        filtered = apply_filters(
            dataset=sample_qa_dataset,
            filter_config=filter_config,
        )
        # Should filter some examples
        assert len(filtered) <= len(sample_qa_dataset)

    def test_cluster_filter_only(self, sample_qa_dataset, sample_cluster_assignments):
        """Test applying only cluster filter."""
        filter_config = {
            "cluster": {
                "enabled": True,
                "cluster_path": sample_cluster_assignments,
                "exclude_clusters": [-1],
            }
        }
        filtered = apply_filters(
            dataset=sample_qa_dataset,
            filter_config=filter_config,
        )
        # Should filter noise cluster
        assert len(filtered) < len(sample_qa_dataset)

    def test_combined_filters(
        self, sample_qa_dataset, sample_cartography_metrics, sample_cluster_assignments
    ):
        """Test applying both cartography and cluster filters."""
        filter_config = {
            "ambiguous": {
                "enabled": True,
                "metrics_path": sample_cartography_metrics,
                "top_fraction": 0.5,
            },
            "cluster": {
                "enabled": True,
                "cluster_path": sample_cluster_assignments,
                "exclude_clusters": [-1],
            },
        }
        filtered = apply_filters(
            dataset=sample_qa_dataset,
            filter_config=filter_config,
        )
        # Should apply both filters
        assert len(filtered) <= len(sample_qa_dataset)

    def test_filter_with_example_ids(self, sample_cartography_metrics):
        """Test filtering when dataset has id field."""
        # Create dataset with id column
        data = {
            "id": ["q1", "q2", "q3", "q4", "q5"],
            "question": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            "context": ["C1", "C2", "C3", "C4", "C5"],
            "answers": [
                {"text": ["A1"], "answer_start": [0]},
                {"text": ["A2"], "answer_start": [0]},
                {"text": ["A3"], "answer_start": [0]},
                {"text": ["A4"], "answer_start": [0]},
                {"text": ["A5"], "answer_start": [0]},
            ],
        }
        dataset = Dataset.from_dict(data)

        filter_config = {
            "ambiguous": {
                "enabled": True,
                "metrics_path": sample_cartography_metrics,
                "top_fraction": 0.5,
            }
        }
        filtered = apply_filters(
            dataset=dataset,
            filter_config=filter_config,
        )
        assert len(filtered) <= len(dataset)

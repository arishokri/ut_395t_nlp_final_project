"""Tests for ID field consistency across the pipeline."""

import pandas as pd
from datasets import Dataset

from cluster_analysis import load_cluster_assignments
from dataset_cartography import load_cartography_metrics
from helpers import generate_hash_ids


class TestIDConsistency:
    """Test that ID fields are handled consistently across the pipeline."""

    def test_generate_hash_ids_returns_id_field(self):
        """Test that generate_hash_ids returns 'id' field."""
        example = {
            "question": "What is the capital?",
            "context": "Paris is the capital of France.",
            "answers": {"text": ["Paris"], "answer_start": [0]},
        }

        result = generate_hash_ids(example)

        assert "id" in result
        assert isinstance(result["id"], str)
        assert len(result["id"]) == 16  # MD5 hash truncated to 16 chars

    def test_dataset_map_adds_id_column(self):
        """Test that dataset.map(generate_hash_ids) adds 'id' column."""
        data = {
            "question": ["Q1?", "Q2?", "Q3?"],
            "context": ["C1", "C2", "C3"],
            "answers": [
                {"text": ["A1"], "answer_start": [0]},
                {"text": ["A2"], "answer_start": [0]},
                {"text": ["A3"], "answer_start": [0]},
            ],
        }
        dataset = Dataset.from_dict(data)

        # Apply hash ID generation
        dataset_with_ids = dataset.map(generate_hash_ids)

        assert "id" in dataset_with_ids.column_names
        assert len(dataset_with_ids["id"]) == 3
        # IDs should be deterministic
        ids_first = dataset_with_ids["id"]
        dataset_with_ids_again = dataset.map(generate_hash_ids)
        assert dataset_with_ids_again["id"] == ids_first

    def test_cartography_metrics_use_example_id_as_index(
        self, sample_cartography_metrics
    ):
        """Test that cartography metrics use 'id' as index (matching dataset and clusters)."""
        df = load_cartography_metrics(sample_cartography_metrics)

        assert df.index.name == "id"
        assert "id" not in df.columns  # Should be index, not column

    def test_cluster_assignments_use_id_as_index(self, sample_cluster_assignments):
        """Test that cluster assignments use 'id' as index."""
        df = load_cluster_assignments(sample_cluster_assignments)

        assert df.index.name == "id"
        assert "id" not in df.columns  # Should be index, not column

    def test_id_mismatch_between_cartography_and_clusters(
        self, sample_cartography_metrics, sample_cluster_assignments
    ):
        """Test that cartography and clusters now use the same index name."""
        cart_df = load_cartography_metrics(sample_cartography_metrics)
        cluster_df = load_cluster_assignments(sample_cluster_assignments)

        # After fix: both should use 'id'
        assert cart_df.index.name == "id"
        assert cluster_df.index.name == "id"
        assert cart_df.index.name == cluster_df.index.name

    def test_merging_cartography_and_cluster_data_requires_rename(
        self, sample_cartography_metrics, sample_cluster_assignments
    ):
        """Test that merging cartography and cluster data works seamlessly with consistent naming."""
        cart_df = load_cartography_metrics(sample_cartography_metrics)
        cluster_df = load_cluster_assignments(sample_cluster_assignments)

        # Reset indices to make them columns
        cart_df = cart_df.reset_index()
        cluster_df = cluster_df.reset_index()

        # Both should now have 'id' column
        assert "id" in cart_df.columns
        assert "id" in cluster_df.columns

        # Can merge directly without renaming
        merged = pd.merge(cart_df, cluster_df, on="id", how="inner")
        assert len(merged) > 0
        assert "id" in merged.columns

    def test_dataset_filtering_expects_id_column(self, sample_qa_dataset):
        """Test that datasets being filtered have 'id' column."""
        # When a dataset is prepared for filtering, it should have 'id' column
        assert "id" in sample_qa_dataset.column_names

        df = sample_qa_dataset.to_pandas()
        assert "id" in df.columns
        assert df["id"].dtype == object  # String IDs

    def test_id_consistency_across_pipeline(
        self, sample_qa_dataset, sample_cartography_metrics, sample_cluster_assignments
    ):
        """Test that IDs can be matched across the full pipeline."""
        # Get dataset IDs
        dataset_ids = set(sample_qa_dataset["id"])

        # Get cartography IDs (now using 'id')
        cart_df = load_cartography_metrics(sample_cartography_metrics)
        cart_ids = set(cart_df.index.tolist())

        # Get cluster IDs (using 'id')
        cluster_df = load_cluster_assignments(sample_cluster_assignments)
        cluster_ids = set(cluster_df.index.tolist())

        # All should refer to the same examples
        assert dataset_ids == cart_ids, "Dataset and cartography IDs should match"
        assert dataset_ids == cluster_ids, "Dataset and cluster IDs should match"
        assert cart_ids == cluster_ids, "Cartography and cluster IDs should match"


class TestIDFieldNaming:
    """Test proper naming conventions for ID fields."""

    def test_preprocessing_converts_id_to_example_id(self):
        """Test that preprocessing converts 'id' to 'example_id' for training."""
        from helpers import prepare_train_dataset_qa
        from transformers import AutoTokenizer

        # Create sample data with 'id' field
        examples = {
            "id": ["id1", "id2"],
            "question": ["Q1?", "Q2?"],
            "context": ["Context 1", "Context 2"],
            "answers": [
                {"text": ["A1"], "answer_start": [0]},
                {"text": ["A2"], "answer_start": [0]},
            ],
        }

        tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

        result = prepare_train_dataset_qa(
            examples, tokenizer=tokenizer, ablations=None, max_seq_length=128
        )

        # Should have 'example_id' field for cartography tracking
        assert "example_id" in result
        assert len(result["example_id"]) >= 2  # May overflow into more features

    def test_example_id_preserved_during_tokenization(self):
        """Test that example_id values match original 'id' values."""
        from helpers import prepare_train_dataset_qa
        from transformers import AutoTokenizer

        examples = {
            "id": ["custom_id_1", "custom_id_2"],
            "question": ["Short question?", "Another short question?"],
            "context": ["Short context.", "Another short context."],
            "answers": [
                {"text": ["context"], "answer_start": [6]},
                {"text": ["context"], "answer_start": [14]},
            ],
        }

        tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

        result = prepare_train_dataset_qa(
            examples, tokenizer=tokenizer, ablations=None, max_seq_length=128
        )

        # example_id should contain the original IDs
        assert "custom_id_1" in result["example_id"]
        assert "custom_id_2" in result["example_id"]

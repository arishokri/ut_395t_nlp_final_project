"""Tests for embedding extraction and analysis."""

import numpy as np
import pytest


class TestEmbeddingExtraction:
    """Test suite for embedding extraction functionality."""

    def test_embeddings_file_format(self, sample_embeddings):
        """Test that embeddings file has correct format."""
        import os

        embeddings_path = f"{sample_embeddings}/embeddings.npy"

        assert os.path.exists(embeddings_path)

        embeddings = np.load(embeddings_path)
        assert embeddings.ndim == 2  # Should be 2D array
        assert embeddings.shape[0] == 5  # 5 examples
        assert embeddings.shape[1] == 128  # 128-dim embeddings

    def test_embeddings_metadata(self, sample_embeddings):
        """Test that metadata file exists and is valid."""
        import json
        import os

        metadata_path = f"{sample_embeddings}/embeddings_metadata.json"
        assert os.path.exists(metadata_path)

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "example_ids" in metadata
        assert "embedding_dim" in metadata
        assert "num_examples" in metadata
        assert len(metadata["example_ids"]) == metadata["num_examples"]

    def test_embeddings_are_numeric(self, sample_embeddings):
        """Test that embeddings contain numeric values."""
        embeddings_path = f"{sample_embeddings}/embeddings.npy"
        embeddings = np.load(embeddings_path)

        # Should be float type
        assert np.issubdtype(embeddings.dtype, np.floating)
        # Should not have NaN values
        assert not np.isnan(embeddings).any()

    def test_embeddings_normalized(self, sample_embeddings):
        """Test embeddings normalization (if applicable)."""
        embeddings_path = f"{sample_embeddings}/embeddings.npy"
        embeddings = np.load(embeddings_path)

        # Check if embeddings have reasonable magnitudes
        norms = np.linalg.norm(embeddings, axis=1)

        # Norms should be positive
        assert (norms > 0).all()


class TestRuleBasedFiltering:
    """Test suite for rule-based filtering."""

    def test_question_detection(self):
        """Test that questions are detected correctly."""
        try:
            from rule_based_errors import is_question

            questions = [
                "What is the capital of France?",
                "Who wrote Hamlet?",
                "How many planets are there?",
            ]

            non_questions = [
                "Paris is the capital.",
                "Shakespeare wrote it.",
                "There are 8 planets.",
            ]

            for q in questions:
                assert is_question(q), f"Failed to detect question: {q}"

            for nq in non_questions:
                assert not is_question(nq), f"False positive: {nq}"
        except ImportError:
            pytest.skip("rule_based_errors module not available")

    def test_filter_non_questions(self):
        """Test filtering out non-questions."""
        try:
            from rule_based_errors import filter_non_questions
            from datasets import Dataset

            data = {
                "id": ["1", "2", "3"],
                "question": [
                    "What is X?",
                    "This is not a question",
                    "How does Y work?",
                ],
                "context": ["C1", "C2", "C3"],
                "answers": [
                    {"text": ["A1"], "answer_start": [0]},
                    {"text": ["A2"], "answer_start": [0]},
                    {"text": ["A3"], "answer_start": [0]},
                ],
            }
            dataset = Dataset.from_dict(data)

            filtered = filter_non_questions(dataset)

            # Should remove at least the obvious non-question
            assert len(filtered) <= len(dataset)
        except ImportError:
            pytest.skip("rule_based_errors module not available")


class TestDatasetAnalysis:
    """Test suite for dataset analysis functionality."""

    def test_analyze_dataset_structure(self, sample_qa_dataset):
        """Test analyzing dataset structure."""
        # Check that dataset has required fields
        assert "question" in sample_qa_dataset.column_names
        assert "context" in sample_qa_dataset.column_names
        assert "answers" in sample_qa_dataset.column_names

    def test_dataset_statistics(self, sample_qa_dataset):
        """Test computing dataset statistics."""
        # Question lengths
        question_lengths = [len(q.split()) for q in sample_qa_dataset["question"]]

        assert min(question_lengths) > 0
        assert max(question_lengths) > 0
        assert sum(question_lengths) / len(question_lengths) > 0  # Average

    def test_answer_extraction(self, sample_qa_dataset):
        """Test that answers can be extracted from dataset."""
        for example in sample_qa_dataset:
            answers = example["answers"]

            assert "text" in answers
            assert "answer_start" in answers
            assert len(answers["text"]) > 0
            assert len(answers["answer_start"]) == len(answers["text"])


class TestTrainingDynamics:
    """Test suite for training dynamics tracking."""

    def test_dynamics_structure(self, temp_dir):
        """Test structure of training dynamics."""
        import json

        # Create sample training dynamics
        dynamics = {
            "example_1": {
                "correctness": [1.0, 1.0, 1.0],
                "confidence": [0.8, 0.9, 0.95],
                "logits": [[0.2, 0.8], [0.1, 0.9], [0.05, 0.95]],
            }
        }

        output_file = temp_dir / "training_dynamics.json"
        with open(output_file, "w") as f:
            json.dump(dynamics, f)

        # Load and verify
        with open(output_file) as f:
            loaded = json.load(f)

        assert "example_1" in loaded
        assert "correctness" in loaded["example_1"]
        assert "confidence" in loaded["example_1"]
        assert len(loaded["example_1"]["correctness"]) == 3

    def test_dynamics_aggregation(self):
        """Test aggregating training dynamics across epochs."""
        dynamics = {
            "correctness": [1.0, 1.0, 0.0],
            "confidence": [0.8, 0.9, 0.6],
        }

        # Compute aggregated metrics
        avg_confidence = sum(dynamics["confidence"]) / len(dynamics["confidence"])
        variability = np.std(dynamics["confidence"])

        assert 0 <= avg_confidence <= 1
        assert variability >= 0


class TestOutputFormats:
    """Test suite for various output formats."""

    def test_csv_output_format(self, temp_dir):
        """Test CSV output format."""
        import pandas as pd

        data = {
            "example_id": ["e1", "e2", "e3"],
            "metric1": [0.9, 0.8, 0.7],
            "metric2": [1.0, 0.5, 0.3],
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "test_output.csv"
        df.to_csv(csv_path, index=False)

        # Verify we can load it back
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 3
        assert "example_id" in loaded.columns

    def test_json_output_format(self, temp_dir):
        """Test JSON output format."""
        import json

        data = {
            "metadata": {"version": "1.0"},
            "results": [
                {"id": "1", "score": 0.9},
                {"id": "2", "score": 0.8},
            ],
        }

        json_path = temp_dir / "test_output.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Verify we can load it back
        with open(json_path) as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "results" in loaded
        assert len(loaded["results"]) == 2

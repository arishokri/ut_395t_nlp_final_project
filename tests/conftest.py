"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_qa_dataset():
    """Create a small sample QA dataset for testing."""
    data = {
        "id": ["q1", "q2", "q3", "q4", "q5"],
        "question": [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What year did World War 2 end?",
            "How many planets are in the solar system?",
            "What is photosynthesis?",
        ],
        "context": [
            "Paris is the capital and most populous city of France.",
            "William Shakespeare wrote many famous plays including Hamlet.",
            "World War 2 ended in 1945 with the surrender of Japan.",
            "The solar system contains 8 planets orbiting the sun.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
        ],
        "answers": [
            {"text": ["Paris"], "answer_start": [0]},
            {"text": ["William Shakespeare"], "answer_start": [0]},
            {"text": ["1945"], "answer_start": [22]},
            {"text": ["8"], "answer_start": [24]},
            {
                "text": ["the process by which plants convert sunlight into energy"],
                "answer_start": [18],
            },
        ],
    }
    return datasets.Dataset.from_dict(data)


@pytest.fixture
def sample_cartography_metrics(temp_dir):
    """Create sample cartography metrics CSV file."""
    metrics_data = {
        "example_id": ["q1", "q2", "q3", "q4", "q5"],
        "confidence": [0.95, 0.85, 0.45, 0.92, 0.50],
        "variability": [0.05, 0.15, 0.55, 0.08, 0.50],
        "correctness": [1.0, 1.0, 0.4, 1.0, 0.6],
        "category": ["easy", "easy", "ambiguous", "easy", "ambiguous"],
    }
    df = pd.DataFrame(metrics_data)

    output_dir = temp_dir / "cartography_output"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "cartography_metrics.csv"
    df.to_csv(csv_path, index=False)

    return str(output_dir)


@pytest.fixture
def sample_cluster_assignments(temp_dir):
    """Create sample cluster assignments CSV file."""
    cluster_data = {
        "id": ["q1", "q2", "q3", "q4", "q5"],
        "cluster": [0, 0, 1, 2, -1],
        "cluster_probability": [0.95, 0.90, 0.85, 0.92, 0.50],
    }
    df = pd.DataFrame(cluster_data)

    output_dir = temp_dir / "cluster_output"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "cluster_assignments.csv"
    df.to_csv(csv_path, index=False)

    return str(output_dir)


@pytest.fixture
def sample_embeddings(temp_dir):
    """Create sample embeddings numpy file."""
    embeddings = np.random.randn(5, 128)  # 5 examples, 128-dim embeddings
    example_ids = ["q1", "q2", "q3", "q4", "q5"]

    output_dir = temp_dir / "embeddings_output"
    output_dir.mkdir(exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "embeddings_metadata.json"

    np.save(embeddings_path, embeddings)

    import json

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "example_ids": example_ids,
                "embedding_dim": 128,
                "num_examples": 5,
            },
            f,
        )

    return str(output_dir)


@pytest.fixture
def mock_model_path():
    """Return a mock model path for testing."""
    return "google/electra-small-discriminator"


@pytest.fixture
def training_args_dict():
    """Return basic training arguments as a dictionary."""
    return {
        "output_dir": "./test_output",
        "do_train": True,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "save_strategy": "no",
        "eval_strategy": "no",
    }

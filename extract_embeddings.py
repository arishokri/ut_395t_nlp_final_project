"""
Embedding Extraction Module for Dataset Analysis

This module extracts embeddings from ELECTRA model representations
to enable clustering and semantic analysis of QA examples.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class EmbeddingExtractor:
    """
    Extracts embeddings from ELECTRA-based QA models.

    Supports extraction of:
    - [CLS] token embeddings from final layer
    - Question-only embeddings
    - Context-only embeddings
    - Combined question-context embeddings
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_path: Path to trained model or HuggingFace model ID
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        print(f"Model loaded on {self.device}")

    def extract_embeddings(
        self,
        questions: List[str],
        contexts: List[str],
        batch_size: int = 32,
        max_length: int = 128,
        embedding_type: str = "cls",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a list of question-context pairs.

        Args:
            questions: List of questions
            contexts: List of contexts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            embedding_type: Type of embedding to extract:
                - 'cls': [CLS] token from final layer
                - 'mean': Mean pooling over all tokens
                - 'question_only': [CLS] from question-only input
                - 'context_only': [CLS] from context-only input
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_examples, embedding_dim)
        """
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must have the same length")

        all_embeddings = []
        n_batches = (len(questions) + batch_size - 1) // batch_size

        iterator = range(0, len(questions), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Extracting embeddings")

        with torch.no_grad():
            for i in iterator:
                batch_questions = questions[i : i + batch_size]
                batch_contexts = contexts[i : i + batch_size]

                if embedding_type == "question_only":
                    embeddings = self._extract_question_embeddings(
                        batch_questions, max_length
                    )
                elif embedding_type == "context_only":
                    embeddings = self._extract_context_embeddings(
                        batch_contexts, max_length
                    )
                else:
                    embeddings = self._extract_combined_embeddings(
                        batch_questions, batch_contexts, max_length, embedding_type
                    )

                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def _get_base_embeddings(
        self,
        texts: List[str],
        text_pairs: Optional[List[str]] = None,
        max_length: int = 128,
        truncation: str = "only_second",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract base embeddings from the model.

        Args:
            texts: Primary text inputs (questions or contexts)
            text_pairs: Optional secondary text inputs (contexts when paired with questions)
            max_length: Maximum sequence length
            truncation: Truncation strategy

        Returns:
            Tuple of (hidden_states, attention_mask)
            - hidden_states: tensor of shape (batch_size, seq_len, hidden_dim)
            - attention_mask: tensor of shape (batch_size, seq_len)
        """
        inputs = self.tokenizer(
            texts,
            text_pairs,
            truncation=truncation if text_pairs else True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings from base model
        if hasattr(self.model, "electra"):
            outputs = self.model.electra(**inputs)
        else:
            outputs = self.model.base_model(**inputs)

        return outputs.last_hidden_state, inputs["attention_mask"]

    def _extract_combined_embeddings(
        self,
        questions: List[str],
        contexts: List[str],
        max_length: int,
        embedding_type: str,
    ) -> np.ndarray:
        """Extract embeddings from question-context pairs."""
        hidden_states, attention_mask = self._get_base_embeddings(
            questions, contexts, max_length, truncation="only_second"
        )

        if embedding_type == "cls":
            # Use [CLS] token (first token)
            embeddings = hidden_states[:, 0, :].cpu().numpy()
        elif embedding_type == "mean":
            # Mean pooling over all tokens (excluding padding)
            attention_mask = attention_mask.unsqueeze(-1)
            masked_embeddings = hidden_states * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        return embeddings

    def _extract_question_embeddings(
        self, questions: List[str], max_length: int
    ) -> np.ndarray:
        """Extract embeddings from questions only."""
        hidden_states, _ = self._get_base_embeddings(questions, None, max_length)
        return hidden_states[:, 0, :].cpu().numpy()

    def _extract_context_embeddings(
        self, contexts: List[str], max_length: int
    ) -> np.ndarray:
        """Extract embeddings from contexts only."""
        hidden_states, _ = self._get_base_embeddings(contexts, None, max_length)
        return hidden_states[:, 0, :].cpu().numpy()


def extract_and_save_embeddings(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    embedding_types: List[str] = ["cls"],
    max_length: int = 128,
):
    """
    Extract embeddings for a dataset and save to disk.

    Args:
        model_path: Path to trained model
        dataset_name: HuggingFace dataset name or path to local file
        output_dir: Directory to save embeddings
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        batch_size: Batch size for extraction
        embedding_types: Types of embeddings to extract
        max_length: Maximum sequence length
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_name}")
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=dataset_name)
        data_split = dataset["train"]
    else:
        dataset = datasets.load_dataset(dataset_name)
        data_split = dataset[split]

    if max_samples:
        data_split = data_split.select(range(min(max_samples, len(data_split))))

    print(f"Processing {len(data_split)} examples")

    # Extract data
    questions = data_split["question"]
    contexts = data_split["context"]
    example_ids = data_split["id"] if "id" in data_split.column_names else None

    # Initialize extractor
    extractor = EmbeddingExtractor(model_path)

    # Extract and save each embedding type
    for emb_type in embedding_types:
        print(f"\nExtracting {emb_type} embeddings...")
        embeddings = extractor.extract_embeddings(
            questions=questions,
            contexts=contexts,
            batch_size=batch_size,
            max_length=max_length,
            embedding_type=emb_type,
        )

        # Save embeddings
        output_file = os.path.join(output_dir, f"embeddings_{emb_type}.npy")
        np.save(output_file, embeddings)
        print(f"Saved {emb_type} embeddings to {output_file}")
        print(f"Shape: {embeddings.shape}")

    # Save metadata
    metadata = {
        "model_path": model_path,
        "dataset_name": dataset_name,
        "split": split,
        "n_examples": len(data_split),
        "embedding_types": embedding_types,
        "max_length": max_length,
        "embedding_dim": embeddings.shape[1],
    }

    if example_ids:
        metadata["example_ids"] = example_ids

    metadata_file = os.path.join(output_dir, "embeddings_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_file}")
    print("=" * 70)


def load_embeddings(
    embedding_dir: str, embedding_type: str = "cls"
) -> Tuple[np.ndarray, Dict]:
    """
    Load embeddings from disk.

    Args:
        embedding_dir: Directory containing saved embeddings
        embedding_type: Type of embedding to load

    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    embedding_file = os.path.join(embedding_dir, f"embeddings_{embedding_type}.npy")
    metadata_file = os.path.join(embedding_dir, "embeddings_metadata.json")

    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embeddings file not found: {embedding_file}")

    embeddings = np.load(embedding_file)

    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    return embeddings, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings from QA model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model or HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Eladio/emrqa-msquad",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings_output",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--embedding_types",
        type=str,
        nargs="+",
        default=["cls"],
        choices=["cls", "mean", "question_only", "context_only"],
        help="Types of embeddings to extract",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    extract_and_save_embeddings(
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        embedding_types=args.embedding_types,
        max_length=args.max_length,
    )

"""
Embedding Extraction Module for Dataset Analysis

This module extracts embeddings from ELECTRA model representations
to enable clustering and semantic analysis of QA examples.

Embedding Design:
    Representation = [CLS of (Q, context)] + mean-pooled answer span tokens

    Why this design?
    - [CLS] gives a global representation of Q+context
    - The pooled span gives a local representation of what was actually labeled as answer
    - Together they capture both global semantic context and local answer alignment
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

    Creates embeddings by concatenating:
    1. [CLS] token from (question, context) encoding
    2. Mean-pooled answer span tokens

    This captures both global semantic context and local answer representation.
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

        # Get embedding dimension from model
        if hasattr(self.model, "electra"):
            self.hidden_size = self.model.electra.config.hidden_size
        else:
            self.hidden_size = self.model.config.hidden_size

        print(f"Hidden size: {self.hidden_size}")
        print(f"Output embedding dimension: {self.hidden_size * 2} ([CLS] + span)")

    def extract_embeddings(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[Dict[str, any]],
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for question-context-answer triplets.

        Args:
            questions: List of questions
            contexts: List of contexts
            answers: List of answer dicts with 'text' and 'answer_start' keys
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_examples, hidden_size * 2)
            Each embedding is [CLS vector (hidden_size) + mean-pooled span (hidden_size)]
        """
        if not (len(questions) == len(contexts) == len(answers)):
            raise ValueError(
                "Questions, contexts, and answers must have the same length"
            )

        all_embeddings = []
        n_batches = (len(questions) + batch_size - 1) // batch_size

        iterator = range(0, len(questions), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Extracting embeddings")

        with torch.no_grad():
            for i in iterator:
                batch_questions = questions[i : i + batch_size]
                batch_contexts = contexts[i : i + batch_size]
                batch_answers = answers[i : i + batch_size]

                embeddings = self._extract_batch_embeddings(
                    batch_questions, batch_contexts, batch_answers, max_length
                )
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def _extract_batch_embeddings(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[Dict[str, any]],
        max_length: int,
    ) -> np.ndarray:
        """
        Extract embeddings for a batch.

        Returns:
            numpy array of shape (batch_size, hidden_size * 2)
        """
        # Tokenize with offsets to track answer span positions
        inputs = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get hidden states from ELECTRA
        if hasattr(self.model, "electra"):
            outputs = self.model.electra(**inputs)
        else:
            outputs = self.model.base_model(**inputs)

        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Extract embeddings for each example in batch
        batch_embeddings = []
        for idx in range(len(questions)):
            # 1. Extract [CLS] token (first token)
            cls_embedding = hidden_states[idx, 0, :].cpu().numpy()  # (hidden_size,)

            # 2. Find answer span tokens and mean-pool them
            # Handle both dict and dict-with-lists formats
            if isinstance(answers[idx]["text"], list):
                answer_text = answers[idx]["text"][0]
                answer_start = answers[idx]["answer_start"][0]
            else:
                answer_text = answers[idx]["text"]
                answer_start = answers[idx]["answer_start"]

            answer_end = answer_start + len(answer_text)

            # Find token positions that overlap with answer span
            # The offset_mapping for context tokens (type_id=1) gives positions relative to context
            span_token_ids = []
            sequence_ids = inputs["token_type_ids"][idx].cpu()

            for token_idx, (char_start, char_end) in enumerate(offset_mapping[idx]):
                # Skip special tokens and question tokens
                if sequence_ids[token_idx] != 1:  # Only process context tokens
                    continue

                # Skip padding tokens (offset is (0, 0))
                if char_start == 0 and char_end == 0:
                    continue

                # Check if token overlaps with answer span
                # Offsets are already relative to context string
                if char_start < answer_end and char_end > answer_start:
                    span_token_ids.append(token_idx)

            # Mean-pool span tokens
            if span_token_ids:
                span_embeddings = hidden_states[
                    idx, span_token_ids, :
                ]  # (n_span_tokens, hidden_size)
                span_embedding = (
                    span_embeddings.mean(dim=0).cpu().numpy()
                )  # (hidden_size,)
            else:
                # Fallback: if no tokens found, use CLS as span embedding
                # This can happen if answer is truncated or outside the context window
                span_embedding = cls_embedding.copy()
                if len(answer_text) > 50:
                    # Only warn for short answers - long ones are likely truncated
                    pass
                elif idx < 5:  # Only warn for first few in batch to avoid spam
                    print(
                        f"Warning: No span tokens found for answer '{answer_text[:50]}...' (using CLS fallback)"
                    )

            # 3. Concatenate [CLS] + span
            combined_embedding = np.concatenate(
                [cls_embedding, span_embedding]
            )  # (hidden_size * 2,)
            batch_embeddings.append(combined_embedding)

        return np.array(batch_embeddings)


def extract_and_save_embeddings(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    max_length: int = 512,
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
        max_length: Maximum sequence length
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("EMBEDDING EXTRACTION")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print()

    print(f"Loading dataset: {dataset_name}")
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=dataset_name)
        data_split = dataset["train"]
    else:
        dataset = datasets.load_dataset(dataset_name)
        data_split = dataset[split]

    if max_samples:
        data_split = data_split.select(range(min(max_samples, len(data_split))))

    print(f"Processing {len(data_split)} examples\n")

    # Extract data
    questions = data_split["question"]
    contexts = data_split["context"]
    answers = data_split["answers"]
    example_ids = data_split["id"] if "id" in data_split.column_names else None

    # Initialize extractor
    extractor = EmbeddingExtractor(model_path)

    # Extract embeddings
    print("\nExtracting [CLS + answer span] embeddings...")
    embeddings = extractor.extract_embeddings(
        questions=questions,
        contexts=contexts,
        answers=answers,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Save embeddings
    output_file = os.path.join(output_dir, "embeddings.npy")
    np.save(output_file, embeddings)
    print(f"\n✓ Saved embeddings to {output_file}")
    print(f"  Shape: {embeddings.shape}")
    print(
        f"  Embedding dimension: {embeddings.shape[1]} ([CLS]:{extractor.hidden_size} + span:{extractor.hidden_size})"
    )

    # Save metadata
    metadata = {
        "model_path": model_path,
        "dataset_name": dataset_name,
        "split": split,
        "n_examples": len(data_split),
        "max_length": max_length,
        "embedding_dim": embeddings.shape[1],
        "hidden_size": extractor.hidden_size,
        "embedding_composition": "[CLS of (Q, context)] + mean-pooled answer span",
    }

    if example_ids:
        metadata["example_ids"] = example_ids

    metadata_file = os.path.join(output_dir, "embeddings_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_file}")
    print("\n" + "=" * 70)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 70 + "\n")


def load_embeddings(embedding_dir: str) -> Tuple[np.ndarray, Dict]:
    """
    Load embeddings from disk.

    Args:
        embedding_dir: Directory containing saved embeddings

    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    embedding_file = os.path.join(embedding_dir, "embeddings.npy")
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

    parser = argparse.ArgumentParser(
        description="Extract [CLS + answer span] embeddings from QA model"
    )
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
        "--max_length",
        type=int,
        default=512,
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
        max_length=args.max_length,
    )

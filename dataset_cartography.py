"""
Dataset Cartography Implementation (Swayamdipta et al. 2020)
https://arxiv.org/abs/2009.10795

This module provides utilities for tracking training dynamics and identifying
easy, hard, and ambiguous examples in a QA dataset.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback


class DatasetCartographyCallback(TrainerCallback):
    """
    Callback to track training dynamics for dataset cartography.

    Tracks for each example across epochs:
    - Confidence: probability assigned to the correct answer span
    - Correctness: whether the prediction was correct

    After training, computes:
    - Mean confidence (average probability across epochs)
    - Variability (std dev of probabilities across epochs)
    - Correctness (fraction of epochs with correct prediction)
    """

    def __init__(self, output_dir: str = "./cartography_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Track dynamics per example: {example_id: {epoch: (start_prob, end_prob, correct)}}
        self.training_dynamics = defaultdict(dict)

        # Track example metadata
        self.example_metadata = {}

    def on_epoch_end(self, args, state, control, **kwargs):
        """Save dynamics at the end of each epoch."""
        epoch = int(state.epoch)

        # Save intermediate results
        dynamics_file = os.path.join(
            self.output_dir, f"training_dynamics_epoch_{epoch}.json"
        )

        # Convert to serializable format
        serializable_dynamics = {
            ex_id: {
                str(ep): {
                    "start_prob": float(vals[0]),
                    "end_prob": float(vals[1]),
                    "correct": bool(vals[2]),
                }
                for ep, vals in epochs.items()
            }
            for ex_id, epochs in self.training_dynamics.items()
        }

        with open(dynamics_file, "w") as f:
            json.dump(serializable_dynamics, f, indent=2)

        print(f"\n[Cartography] Saved training dynamics for epoch {epoch}")
        print(f"[Cartography] Tracking {len(self.training_dynamics)} examples")

    def on_train_end(self, args, state, control, **kwargs):
        """Compute final cartography metrics and save results."""
        print("\n[Cartography] Computing final metrics...")

        metrics = self.compute_cartography_metrics()

        # Save metrics as JSON
        metrics_file = os.path.join(self.output_dir, "cartography_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save as DataFrame for analysis
        df = pd.DataFrame.from_dict(metrics, orient="index")
        df.index.name = "example_id"
        df_file = os.path.join(self.output_dir, "cartography_metrics.csv")
        df.to_csv(df_file)

        print(f"[Cartography] Saved metrics to {metrics_file}")
        print(f"[Cartography] Saved DataFrame to {df_file}")

        # Generate summary statistics
        self.print_summary(df)

        # Generate visualization
        self.visualize_cartography(df)

    def compute_cartography_metrics(self) -> Dict:
        """
        Compute cartography metrics for each example.

        Returns:
            Dictionary mapping example_id to metrics:
            - confidence: mean probability across epochs
            - variability: std dev of probabilities across epochs
            - correctness: fraction of correct predictions
            - num_epochs: number of epochs tracked
        """
        metrics = {}

        for example_id, epoch_data in self.training_dynamics.items():
            # Extract probabilities and correctness across epochs
            start_probs = []
            end_probs = []
            correctness = []

            for epoch in sorted(epoch_data.keys()):
                start_prob, end_prob, correct = epoch_data[epoch]
                start_probs.append(start_prob)
                end_probs.append(end_prob)
                correctness.append(1.0 if correct else 0.0)

            # Compute combined probability (geometric mean of start and end)
            combined_probs = [np.sqrt(s * e) for s, e in zip(start_probs, end_probs)]

            metrics[example_id] = {
                "confidence": float(np.mean(combined_probs)),
                "variability": float(np.std(combined_probs)),
                "correctness": float(np.mean(correctness)),
                "num_epochs": len(epoch_data),
                "start_confidence": float(np.mean(start_probs)),
                "end_confidence": float(np.mean(end_probs)),
            }

        return metrics

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics of cartography metrics."""
        print("\n" + "=" * 70)
        print("DATASET CARTOGRAPHY SUMMARY")
        print("=" * 70)

        print(f"\nTotal examples tracked: {len(df)}")

        if len(df) == 0:
            print("\n[WARNING] No examples were tracked during training!")
            print(
                "This likely means 'example_id' was not available in the batch inputs."
            )
            print("Check that the dataset has an 'id' field and it's being preserved.")
            return

        print("\nMetric Statistics:")
        print(df[["confidence", "variability", "correctness"]].describe())

        # Use categorize_examples to classify examples
        df_categorized = categorize_examples(df)
        category_counts = df_categorized["category"].value_counts()

        print(f"\n{'=' * 70}")
        print("EXAMPLE CATEGORIZATION (using median thresholds):")
        print(f"{'=' * 70}")
        print(
            f"Easy to learn (high conf, low var):    {category_counts.get('easy', 0):6d} ({100 * category_counts.get('easy', 0) / len(df):5.1f}%)"
        )
        print(
            f"Hard to learn (low conf, low var):     {category_counts.get('hard', 0):6d} ({100 * category_counts.get('hard', 0) / len(df):5.1f}%)"
        )
        print(
            f"Ambiguous (low conf, high var):        {category_counts.get('ambiguous', 0):6d} ({100 * category_counts.get('ambiguous', 0) / len(df):5.1f}%)"
        )
        print("=" * 70 + "\n")

    def visualize_cartography(self, df: pd.DataFrame):
        """Create data map visualization (confidence vs variability)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Confidence vs Variability (classic data map)
        ax = axes[0]
        scatter = ax.scatter(
            df["variability"],
            df["confidence"],
            c=df["correctness"],
            cmap="RdYlGn",
            alpha=0.6,
            s=20,
        )
        ax.set_xlabel("Variability", fontsize=12)
        ax.set_ylabel("Confidence", fontsize=12)
        ax.set_title("Dataset Cartography Map", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add median lines
        ax.axhline(
            df["confidence"].median(),
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Median confidence",
        )
        ax.axvline(
            df["variability"].median(),
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Median variability",
        )
        ax.legend(fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Correctness", fontsize=10)

        # Plot 2: Distribution of categories
        ax = axes[1]

        # Use categorize_examples to classify examples
        df_categorized = categorize_examples(df)
        category_counts = df_categorized["category"].value_counts()

        # Get counts for each category (use .get() for safe access)
        easy = category_counts.get("easy", 0)
        hard = category_counts.get("hard", 0)
        ambiguous = category_counts.get("ambiguous", 0)

        categories = ["Easy", "Hard", "Ambiguous"]
        counts = [easy, hard, ambiguous]
        colors = ["green", "red", "orange"]

        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_ylabel("Number of Examples", fontsize=12)
        ax.set_title("Example Distribution by Category", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}\n({100 * count / len(df):.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.output_dir, "cartography_map.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[Cartography] Saved visualization to {fig_path}")

        plt.close()

    def record_batch_dynamics(
        self,
        example_ids: List[str],
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        epoch: int,
    ):
        """
        Record training dynamics for a batch of examples.

        Args:
            example_ids: List of example IDs
            start_logits: Model's start position logits [batch_size, seq_len]
            end_logits: Model's end position logits [batch_size, seq_len]
            start_positions: True start positions [batch_size]
            end_positions: True end positions [batch_size]
            epoch: Current epoch number
        """
        # Convert to probabilities
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        # Get predictions
        start_preds = torch.argmax(start_logits, dim=-1)
        end_preds = torch.argmax(end_logits, dim=-1)

        # Move to CPU for processing
        start_probs = start_probs.cpu().detach()
        end_probs = end_probs.cpu().detach()
        start_preds = start_preds.cpu().detach()
        end_preds = end_preds.cpu().detach()
        start_positions = start_positions.cpu().detach()
        end_positions = end_positions.cpu().detach()

        for i, example_id in enumerate(example_ids):
            # Get probability assigned to correct answer
            true_start_pos = start_positions[i].item()
            true_end_pos = end_positions[i].item()

            start_prob = start_probs[i, true_start_pos].item()
            end_prob = end_probs[i, true_end_pos].item()

            # Check if prediction is correct
            pred_start = start_preds[i].item()
            pred_end = end_preds[i].item()
            correct = (pred_start == true_start_pos) and (pred_end == true_end_pos)

            # Store dynamics
            self.training_dynamics[example_id][epoch] = (start_prob, end_prob, correct)


def load_cartography_metrics(output_dir: str = "./cartography_output") -> pd.DataFrame:
    """
    Load cartography metrics from a completed training run.

    Args:
        output_dir: Directory containing cartography output files

    Returns:
        DataFrame with cartography metrics for each example
    """
    metrics_file = os.path.join(output_dir, "cartography_metrics.csv")

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(
            f"Cartography metrics file not found: {metrics_file}\n"
            "Make sure to run training with --enable_cartography flag first."
        )

    df = pd.read_csv(metrics_file, index_col="example_id")
    return df


def load_variability_map(
    cartography_dir: str,
    mode: str = "smoothing",
    smoothing_factor: float = 0.6,
    weight_clip_range: tuple = (0.1, 10.0),
) -> dict:
    """
    Load variability scores from cartography metrics for label smoothing or loss weighting.

    Supports two modes:
    - 'smoothing': Converts variability to label smoothing factors (0.0-0.3 range)
    - 'weighting': Converts variability to loss weights (1.0+ range, clipped)

    Args:
        cartography_dir: Directory containing cartography_metrics.csv
        mode: Either 'smoothing' or 'weighting'
        smoothing_factor: For mode='smoothing', multiplier to convert variability
                         to smoothing amount (default: 0.6)
        weight_clip_range: For mode='weighting', (min, max) range to clip weights
                          (default: (0.1, 10.0))

    Returns:
        Dictionary mapping example_id (str) -> value (float)
        - For 'smoothing': value is smoothing amount in [0.0, 0.3]
        - For 'weighting': value is loss weight in [clip_min, clip_max]

    Example:
        # For label smoothing
        smoothing_map = load_variability_map('./cartography_output', mode='smoothing')
        smoothing = smoothing_map['abc123']  # Returns 0.15 for var=0.25

        # For loss weighting
        weight_map = load_variability_map('./cartography_output', mode='weighting')
        weight = weight_map['abc123']  # Returns 1.25 for var=0.25
    """
    df = load_cartography_metrics(cartography_dir)

    result_map = {}

    if mode == "smoothing":
        # Convert variability to smoothing factors
        # Higher variability → more smoothing (softer targets)
        for example_id, row in df.iterrows():
            variability = row["variability"]
            # Typical variability range: 0.0 - 0.4
            # Desired smoothing range: 0.0 - 0.3
            smoothing = min(0.3, variability * smoothing_factor)
            result_map[example_id] = float(smoothing)

    elif mode == "weighting":
        # Convert variability to loss weights
        # Higher variability → higher weight (focus more on hard examples)
        for example_id, row in df.iterrows():
            variability = row["variability"]
            raw_weight = 1.0 + variability  # Range: [1.0, ~2.5] for typical variability
            # Clip to prevent extreme weights
            weight = max(weight_clip_range[0], min(weight_clip_range[1], raw_weight))
            result_map[example_id] = float(weight)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'smoothing' or 'weighting'.")

    return result_map


def categorize_examples(
    df: pd.DataFrame,
    conf_threshold: Optional[float] = None,
    var_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Categorize examples as easy, hard, or ambiguous.

    Args:
        df: DataFrame with cartography metrics
        conf_threshold: Confidence threshold (default: median)
        var_threshold: Variability threshold (default: median)

    Returns:
        DataFrame with added 'category' column
    """
    if conf_threshold is None:
        conf_threshold = df["confidence"].median()
    if var_threshold is None:
        var_threshold = df["variability"].median()

    def categorize(row):
        if row["variability"] <= var_threshold:
            if row["confidence"] >= conf_threshold:
                return "easy"
            else:
                return "hard"
        else:
            return "ambiguous"
        # if row["confidence"] >= conf_threshold:
        #     if row["variability"] <= var_threshold:
        #         return "easy"
        #     else:
        #         return "easy_variable"  # high conf, high var
        # else:
        #     if row["variability"] <= var_threshold:
        #         return "hard"
        #     else:
        #         return "ambiguous"

    df["category"] = df.apply(categorize, axis=1)
    return df


def get_examples_by_category(df: pd.DataFrame, category: str, n: int = 10) -> List[str]:
    """
    Get example IDs from a specific category.

    Args:
        df: DataFrame with cartography metrics and categories
        category: One of 'easy', 'hard', 'ambiguous', 'easy_variable'
        n: Number of examples to return

    Returns:
        List of example IDs
    """
    if "category" not in df.columns:
        df = categorize_examples(df)

    category_examples = df[df["category"] == category]

    # Sort by confidence (ascending for hard/ambiguous, descending for easy)
    if category in ["hard", "ambiguous"]:
        category_examples = category_examples.sort_values("confidence", ascending=True)
    else:
        category_examples = category_examples.sort_values("confidence", ascending=False)

    return category_examples.head(n).index.tolist()


def analyze_cartography_by_question_type(
    cartography_df: pd.DataFrame, dataset_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze cartography metrics by question type.

    Args:
        cartography_df: DataFrame with cartography metrics (indexed by example_id)
        dataset_df: DataFrame with dataset examples including 'question' column

    Returns:
        DataFrame with metrics grouped by question type
    """
    # Check if dataset has 'id' column
    if "id" not in dataset_df.columns:
        raise ValueError(
            "dataset_df must have an 'id' column. "
            "Make sure to generate IDs using generate_hash_ids() if needed."
        )

    # Merge cartography metrics with dataset
    # Only keep questions for examples we have cartography data for
    merged = cartography_df.join(dataset_df.set_index("id")[["question"]], how="inner")

    # Classify question types
    def classify_question(q):
        q_low = q.lower()
        if any(x in q_low for x in ["when", "date", "time", "year"]):
            return "temporal"
        elif any(x in q_low for x in ["how many", "how much", "dose", "dosage"]):
            return "numerical"
        elif any(x in q_low for x in ["what", "which"]):
            return "what/which"
        elif any(x in q_low for x in ["has", "does", "is", "was", "did"]):
            return "yes/no"
        elif any(x in q_low for x in ["why"]):
            return "why"
        elif any(x in q_low for x in ["how"]):
            return "how"
        else:
            return "other"

    merged["question_type"] = merged["question"].apply(classify_question)

    # Group by question type and compute statistics
    grouped = (
        merged.groupby("question_type")
        .agg(
            {
                "confidence": ["mean", "std", "count"],
                "variability": ["mean", "std"],
                "correctness": ["mean", "std"],
            }
        )
        .round(3)
    )

    return grouped

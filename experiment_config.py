"""
Experiment Configuration System

This module defines experiment configurations for systematic training runs
with different filtering and training strategies.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class CartographyFilterConfig:
    """Configuration for cartography-based filtering."""

    enabled: bool = False
    cartography_output_dir: str = "./cartography_output"
    top_fraction: float = 0.33  # Keep top 33% most ambiguous
    apply_rule_based_filter: bool = False


@dataclass
class ClusterFilterConfig:
    """Configuration for cluster-based filtering."""

    enabled: bool = False
    cluster_assignments_path: Optional[str] = None
    exclude_clusters: List[int] = field(default_factory=lambda: [-1])
    min_cluster_probability: Optional[float] = None


@dataclass
class LabelSmoothingConfig:
    """Configuration for variability-based label smoothing."""

    enabled: bool = False
    cartography_output_dir: str = "./cartography_output"
    smoothing_factor: float = 0.6


@dataclass
class SoftWeightingConfig:
    """Configuration for soft weight schedule using variability."""

    enabled: bool = False
    cartography_output_dir: str = "./cartography_output"
    weight_clip_min: float = 0.1
    weight_clip_max: float = 10.0


@dataclass
class TrainingConfig:
    """Base training configuration."""

    # Model and dataset
    model: str = "google/electra-small-discriminator"
    dataset: str = "Eladio/emrqa-msquad"
    max_length: int = 512
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Training hyperparameters
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    seed: int = 42

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"

    # Ablations
    ablations: str = "none"  # 'none', 'q_only', 'p_only'

    # Filtering strategies
    cartography_filter: CartographyFilterConfig = field(
        default_factory=CartographyFilterConfig
    )
    cluster_filter: ClusterFilterConfig = field(default_factory=ClusterFilterConfig)

    # Training strategies
    label_smoothing: LabelSmoothingConfig = field(default_factory=LabelSmoothingConfig)
    soft_weighting: SoftWeightingConfig = field(default_factory=SoftWeightingConfig)

    # Cartography tracking for validation set
    enable_cartography: bool = False
    cartography_output_dir: str = "./cartography_output"

    # Additional flags
    filter_validation_set: bool = (
        False  # Whether to apply same filters to validation set
    )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str  # Experiment name
    description: str  # What this experiment tests
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./experiments"  # Base output directory

    def __post_init__(self):
        """Generate experiment hash for unique identification."""
        # Create a unique hash based on configuration
        config_str = json.dumps(asdict(self), sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_output_dir(self) -> str:
        """Get the full output directory path for this experiment."""
        # Create directory name: experiments/<name>_<hash>
        dir_name = f"{self.name}_{self.config_hash}"
        return os.path.join(self.output_dir, dir_name)

    def get_trainer_output_dir(self) -> str:
        """Get the trainer output directory."""
        return os.path.join(self.get_output_dir(), "trainer_output")

    def get_results_file(self) -> str:
        """Get path to results JSON file."""
        return os.path.join(self.get_output_dir(), "results.json")

    def get_config_file(self) -> str:
        """Get path to config JSON file."""
        return os.path.join(self.get_output_dir(), "config.json")

    def save(self):
        """Save experiment configuration to disk."""
        os.makedirs(self.get_output_dir(), exist_ok=True)

        config_dict = asdict(self)
        config_dict["config_hash"] = self.config_hash

        with open(self.get_config_file(), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, config_file: str) -> "ExperimentConfig":
        """Load experiment configuration from disk."""
        with open(config_file, "r") as f:
            config_dict = json.load(f)

        # Remove hash from dict (will be regenerated)
        config_dict.pop("config_hash", None)

        # Reconstruct nested dataclasses
        training_dict = config_dict.pop("training")

        # Reconstruct filter/strategy configs
        cartography_filter = CartographyFilterConfig(
            **training_dict.pop("cartography_filter")
        )
        cluster_filter = ClusterFilterConfig(**training_dict.pop("cluster_filter"))
        label_smoothing = LabelSmoothingConfig(**training_dict.pop("label_smoothing"))
        soft_weighting = SoftWeightingConfig(**training_dict.pop("soft_weighting"))

        training = TrainingConfig(
            **training_dict,
            cartography_filter=cartography_filter,
            cluster_filter=cluster_filter,
            label_smoothing=label_smoothing,
            soft_weighting=soft_weighting,
        )

        return cls(**config_dict, training=training)

    def to_cli_args(self) -> List[str]:
        """
        Convert experiment config to command-line arguments for run.py.

        Returns:
            List of CLI arguments
        """
        args = []

        # Training mode
        args.extend(["--do_train", "--do_eval"])

        # Model and dataset
        args.extend(["--model", self.training.model])
        args.extend(["--dataset", self.training.dataset])
        args.extend(["--max_length", str(self.training.max_length)])

        if self.training.max_train_samples:
            args.extend(["--max_train_samples", str(self.training.max_train_samples)])
        if self.training.max_eval_samples:
            args.extend(["--max_eval_samples", str(self.training.max_eval_samples)])

        # Training hyperparameters
        args.extend(
            [
                "--per_device_train_batch_size",
                str(self.training.per_device_train_batch_size),
            ]
        )
        args.extend(
            [
                "--per_device_eval_batch_size",
                str(self.training.per_device_eval_batch_size),
            ]
        )
        args.extend(["--num_train_epochs", str(self.training.num_train_epochs)])
        args.extend(["--learning_rate", str(self.training.learning_rate)])
        args.extend(["--weight_decay", str(self.training.weight_decay)])
        args.extend(["--warmup_steps", str(self.training.warmup_steps)])
        args.extend(["--seed", str(self.training.seed)])

        # Evaluation and saving
        args.extend(["--eval_steps", str(self.training.eval_steps)])
        args.extend(["--save_steps", str(self.training.save_steps)])
        args.extend(["--logging_steps", str(self.training.logging_steps)])
        args.extend(["--evaluation_strategy", self.training.evaluation_strategy])
        args.extend(["--save_strategy", self.training.save_strategy])
        args.extend(["--save_total_limit", str(self.training.save_total_limit)])
        args.extend(["--metric_for_best_model", self.training.metric_for_best_model])

        if self.training.load_best_model_at_end:
            args.append("--load_best_model_at_end")

        # Output directory
        args.extend(["--output_dir", self.get_trainer_output_dir()])

        # Ablations
        args.extend(["--ablations", self.training.ablations])

        # Cartography filtering
        if self.training.cartography_filter.enabled:
            args.append("--filter_cartography")
            args.extend(
                [
                    "--cartography_output_dir",
                    self.training.cartography_filter.cartography_output_dir,
                ]
            )

        # Cluster filtering
        if self.training.cluster_filter.enabled:
            args.append("--filter_clusters")
            if self.training.cluster_filter.cluster_assignments_path:
                args.extend(
                    [
                        "--cluster_assignments_path",
                        self.training.cluster_filter.cluster_assignments_path,
                    ]
                )
            if self.training.cluster_filter.exclude_clusters:
                exclude_str = ",".join(
                    map(str, self.training.cluster_filter.exclude_clusters)
                )
                args.extend(["--exclude_clusters", exclude_str])
            if self.training.cluster_filter.min_cluster_probability is not None:
                args.extend(
                    [
                        "--min_cluster_probability",
                        str(self.training.cluster_filter.min_cluster_probability),
                    ]
                )

        # Label smoothing
        if self.training.label_smoothing.enabled:
            args.append("--use_label_smoothing")
            args.extend(
                [
                    "--smoothing_factor",
                    str(self.training.label_smoothing.smoothing_factor),
                ]
            )
            args.extend(
                [
                    "--cartography_output_dir",
                    self.training.label_smoothing.cartography_output_dir,
                ]
            )

        # Soft weighting
        if self.training.soft_weighting.enabled:
            args.append("--use_soft_weighting")
            args.extend(
                ["--weight_clip_min", str(self.training.soft_weighting.weight_clip_min)]
            )
            args.extend(
                ["--weight_clip_max", str(self.training.soft_weighting.weight_clip_max)]
            )
            args.extend(
                [
                    "--cartography_output_dir",
                    self.training.soft_weighting.cartography_output_dir,
                ]
            )

        # Cartography tracking
        if self.training.enable_cartography:
            args.append("--enable_cartography")
            args.extend(
                ["--cartography_output_dir", self.training.cartography_output_dir]
            )

        return args


def create_baseline_config(
    name: str = "baseline",
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> ExperimentConfig:
    """Create a baseline experiment configuration."""
    training = TrainingConfig(
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    return ExperimentConfig(
        name=name,
        description="Baseline training without any filtering or special strategies",
        training=training,
    )


def create_cartography_filter_config(
    name: str = "cartography_filter",
    cartography_dir: str = "./cartography_output",
    top_fraction: float = 0.33,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> ExperimentConfig:
    """Create an experiment with cartography-based filtering."""
    cartography_filter = CartographyFilterConfig(
        enabled=True,
        cartography_output_dir=cartography_dir,
        top_fraction=top_fraction,
    )

    training = TrainingConfig(
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        cartography_filter=cartography_filter,
    )

    return ExperimentConfig(
        name=name,
        description=f"Training with cartography filtering (top {top_fraction:.0%} ambiguous kept)",
        training=training,
    )


def create_cluster_filter_config(
    name: str = "cluster_filter",
    cluster_path: str = "./cluster_output",
    exclude_clusters: List[int] = None,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> ExperimentConfig:
    """Create an experiment with cluster-based filtering."""
    if exclude_clusters is None:
        exclude_clusters = [-1]  # Default: exclude noise

    cluster_filter = ClusterFilterConfig(
        enabled=True,
        cluster_assignments_path=cluster_path,
        exclude_clusters=exclude_clusters,
    )

    training = TrainingConfig(
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        cluster_filter=cluster_filter,
    )

    return ExperimentConfig(
        name=name,
        description=f"Training with cluster filtering (excluding clusters {exclude_clusters})",
        training=training,
    )


def create_label_smoothing_config(
    name: str = "label_smoothing",
    cartography_dir: str = "./cartography_output",
    smoothing_factor: float = 0.6,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> ExperimentConfig:
    """Create an experiment with variability-based label smoothing."""
    label_smoothing = LabelSmoothingConfig(
        enabled=True,
        cartography_output_dir=cartography_dir,
        smoothing_factor=smoothing_factor,
    )

    training = TrainingConfig(
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        label_smoothing=label_smoothing,
    )

    return ExperimentConfig(
        name=name,
        description=f"Training with variability-based label smoothing (factor={smoothing_factor})",
        training=training,
    )


def create_soft_weighting_config(
    name: str = "soft_weighting",
    cartography_dir: str = "./cartography_output",
    weight_clip_min: float = 0.1,
    weight_clip_max: float = 10.0,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> ExperimentConfig:
    """Create an experiment with soft weighting based on variability."""
    soft_weighting = SoftWeightingConfig(
        enabled=True,
        cartography_output_dir=cartography_dir,
        weight_clip_min=weight_clip_min,
        weight_clip_max=weight_clip_max,
    )

    training = TrainingConfig(
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        soft_weighting=soft_weighting,
    )

    return ExperimentConfig(
        name=name,
        description=f"Training with soft weighting (clip range [{weight_clip_min}, {weight_clip_max}])",
        training=training,
    )

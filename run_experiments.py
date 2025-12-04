#!/usr/bin/env python3
"""
Run Experiments

Main script for running systematic experiments with different configurations.
"""

import argparse
import sys

from experiment_analysis import analyze_all_experiments
from experiment_config import (
    create_baseline_config,
    create_cartography_filter_config,
    create_cluster_filter_config,
    create_label_smoothing_config,
    create_soft_weighting_config,
)
from experiment_runner import run_experiments_from_list


def create_sample_experiment_suite(
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
    cartography_dir: str = "./cartography_output",
    validation_cartography_dir: str = "./cartography_output_validation",
    cluster_dir: str = "./cluster_output",
    seeds: list = None,
) -> list:
    """
    Create a sample suite of experiments.

    Args:
        max_train_samples: Maximum training samples to use
        max_eval_samples: Maximum evaluation samples to use
        cartography_dir: Directory with cartography metrics for training
        validation_cartography_dir: Directory with cartography metrics for validation
        cluster_dir: Directory with cluster assignments
        seeds: List of random seeds to use (for replication)

    Returns:
        List of experiment configurations
    """
    if seeds is None:
        seeds = [42]  # Default: single seed

    experiments = []

    # For each seed, create different experiment configurations
    for seed in seeds:
        seed_suffix = f"_seed{seed}"

        # 1. Baseline
        experiments.append(
            create_baseline_config(
                name=f"baseline{seed_suffix}",
                seed=seed,
                max_train_samples=max_train_samples,
                max_eval_samples=max_eval_samples,
            )
        )

        # 2. Cartography filtering (different fractions)
        for fraction in [0.25, 0.33, 0.50]:
            experiments.append(
                create_cartography_filter_config(
                    name=f"cartography_top{int(fraction * 100)}{seed_suffix}",
                    cartography_dir=cartography_dir,
                    top_fraction=fraction,
                    seed=seed,
                    max_train_samples=max_train_samples,
                    max_eval_samples=max_eval_samples,
                )
            )

        # 3. Cluster filtering
        experiments.append(
            create_cluster_filter_config(
                name=f"cluster_filter{seed_suffix}",
                cluster_path=cluster_dir,
                exclude_clusters=[-1],  # Exclude noise
                seed=seed,
                max_train_samples=max_train_samples,
                max_eval_samples=max_eval_samples,
            )
        )

        # 4. Label smoothing (different factors)
        for factor in [0.4, 0.6, 0.8]:
            experiments.append(
                create_label_smoothing_config(
                    name=f"label_smooth{int(factor * 10)}{seed_suffix}",
                    cartography_dir=cartography_dir,
                    smoothing_factor=factor,
                    seed=seed,
                    max_train_samples=max_train_samples,
                    max_eval_samples=max_eval_samples,
                )
            )

        # 5. Soft weighting
        experiments.append(
            create_soft_weighting_config(
                name=f"soft_weight{seed_suffix}",
                cartography_dir=cartography_dir,
                seed=seed,
                max_train_samples=max_train_samples,
                max_eval_samples=max_eval_samples,
            )
        )

    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sample experiment suite
  python run_experiments.py --mode run --suite sample

  # Analyze existing results
  python run_experiments.py --mode analyze

  # Dry run to preview experiments
  python run_experiments.py --mode run --suite sample --dry-run

  # Run with custom parameters
  python run_experiments.py --mode run --suite sample \\
      --max-train-samples 5000 \\
      --seeds 42 43 44

  # Run experiments from saved configs
  python run_experiments.py --mode run --configs experiments/*/config.json
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "analyze"],
        required=True,
        help="Mode: 'run' experiments or 'analyze' results",
    )

    # Run mode arguments
    parser.add_argument(
        "--suite",
        type=str,
        choices=["sample", "minimal", "filtering", "weighting"],
        help="Experiment suite to run (for --mode run)",
    )

    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Paths to experiment config files (for --mode run)",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Maximum training samples (default: 10000)",
    )

    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=2000,
        help="Maximum evaluation samples (default: 2000)",
    )

    parser.add_argument(
        "--cartography-dir",
        type=str,
        default="./cartography_output",
        help="Directory with cartography metrics for training (default: ./cartography_output)",
    )

    parser.add_argument(
        "--validation-cartography-dir",
        type=str,
        default="./cartography_output_validation",
        help="Directory with cartography metrics for validation (default: ./cartography_output_validation)",
    )

    parser.add_argument(
        "--cluster-dir",
        type=str,
        default="./cluster_output",
        help="Directory with cluster assignments (default: ./cluster_output)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for replication (default: 42)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on first error (default: continue)",
    )

    # Analyze mode arguments
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="./experiments",
        help="Directory containing experiments (default: ./experiments)",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (default: save only)",
    )

    args = parser.parse_args()

    if args.mode == "run":
        # Create experiments
        if args.suite == "sample":
            experiments = create_sample_experiment_suite(
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                cartography_dir=args.cartography_dir,
                validation_cartography_dir=args.validation_cartography_dir,
                cluster_dir=args.cluster_dir,
                seeds=args.seeds,
            )
            print(f"\n[INFO] Created {len(experiments)} experiments from sample suite")

        elif args.suite == "minimal":
            # Minimal suite for quick testing
            experiments = [
                create_baseline_config(
                    name="baseline",
                    seed=42,
                    max_train_samples=args.max_train_samples,
                    max_eval_samples=args.max_eval_samples,
                ),
                #! Minimal does not filter the validation set on evaluation.
                create_cartography_filter_config(
                    name="cartography_filter",
                    cartography_dir=args.cartography_dir,
                    seed=42,
                    max_train_samples=args.max_train_samples,
                    max_eval_samples=args.max_eval_samples,
                ),
            ]
            print(f"\n[INFO] Created {len(experiments)} experiments from minimal suite")

        elif args.suite == "filtering":
            # Filtering suite: test filtering methods only (with validation filtering)
            from experiment_config import (
                TrainingConfig,
                CartographyFilterConfig,
                ClusterFilterConfig,
                ExperimentConfig,
            )

            experiments = [
                # 1. Baseline
                create_baseline_config(
                    name="baseline",
                    seed=42,
                    max_train_samples=args.max_train_samples,
                    max_eval_samples=args.max_eval_samples,
                ),
                # 2. Cartography filtering (with validation filtering)
                ExperimentConfig(
                    name="cartography_filter",
                    description="Cartography filtering on training + validation",
                    training=TrainingConfig(
                        seed=42,
                        max_train_samples=args.max_train_samples,
                        max_eval_samples=args.max_eval_samples,
                        cartography_filter=CartographyFilterConfig(
                            enabled=True,
                            cartography_output_dir=args.cartography_dir,
                            top_fraction=0.33,
                        ),
                        filter_validation_set=True,
                        validation_cartography_output_dir=args.validation_cartography_dir,
                    ),
                ),
                # 3. Cluster filtering (with validation filtering)
                ExperimentConfig(
                    name="cluster_filter",
                    description="Cluster filtering on training + validation",
                    training=TrainingConfig(
                        seed=42,
                        max_train_samples=args.max_train_samples,
                        max_eval_samples=args.max_eval_samples,
                        cluster_filter=ClusterFilterConfig(
                            enabled=True,
                            cluster_assignments_path=args.cluster_dir,
                            exclude_clusters=[-1],
                        ),
                        filter_validation_set=True,
                        validation_cluster_assignments_path="./cluster_output_validation",
                    ),
                ),
            ]
            print(
                f"\n[INFO] Created {len(experiments)} experiments from filtering suite"
            )
            print(
                "[INFO] Validation filtering is ENABLED for all filtering experiments"
            )

        elif args.suite == "weighting":
            # Weighting suite: test weighting and smoothing methods
            experiments = [
                # 1. Baseline
                create_baseline_config(
                    name="baseline",
                    seed=42,
                    max_train_samples=args.max_train_samples,
                    max_eval_samples=args.max_eval_samples,
                ),
                # 2. Label smoothing
                create_label_smoothing_config(
                    name="label_smoothing",
                    cartography_dir=args.cartography_dir,
                    smoothing_factor=0.6,
                    seed=42,
                    max_train_samples=args.max_train_samples,
                    max_eval_samples=args.max_eval_samples,
                ),
                # 3. Soft weighting
                create_soft_weighting_config(
                    name="soft_weighting",
                    cartography_dir=args.cartography_dir,
                    seed=42,
                    max_train_samples=args.max_train_samples,
                    max_eval_samples=args.max_eval_samples,
                ),
            ]
            print(
                f"\n[INFO] Created {len(experiments)} experiments from weighting suite"
            )

        elif args.configs:
            # Load from config files
            from experiment_runner import run_experiments_from_configs

            run_experiments_from_configs(
                config_files=args.configs,
                dry_run=args.dry_run,
                continue_on_error=not args.stop_on_error,
            )
            return

        else:
            print("[ERROR] Must specify --suite or --configs for run mode")
            sys.exit(1)

        # Run experiments
        run_experiments_from_list(
            experiments=experiments,
            dry_run=args.dry_run,
            continue_on_error=not args.stop_on_error,
        )

    elif args.mode == "analyze":
        # Analyze results
        analyze_all_experiments(
            experiments_dir=args.experiments_dir,
            show_plots=args.show_plots,
        )


if __name__ == "__main__":
    main()

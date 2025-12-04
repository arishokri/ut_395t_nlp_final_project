"""
Experiment Runner

This module provides utilities for running multiple training experiments
and tracking their results.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from experiment_config import ExperimentConfig


class ExperimentResult:
    """Container for experiment results."""

    def __init__(
        self,
        config: ExperimentConfig,
        success: bool = False,
        metrics: Optional[Dict] = None,
        error: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        self.config = config
        self.success = success
        self.metrics = metrics or {}
        self.error = error
        self.start_time = start_time
        self.end_time = end_time

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.config.name,
            "config_hash": self.config.config_hash,
            "success": self.success,
            "metrics": self.metrics,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat()
            if self.start_time
            else None,
        }

    def save(self):
        """Save result to experiment directory."""
        results_file = self.config.get_results_file()
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentRunner:
    """Runner for executing multiple experiments."""

    def __init__(
        self,
        experiments: List[ExperimentConfig],
        dry_run: bool = False,
        continue_on_error: bool = True,
    ):
        """
        Initialize experiment runner.

        Args:
            experiments: List of experiment configurations to run
            dry_run: If True, only print commands without executing
            continue_on_error: If True, continue running experiments even if one fails
        """
        self.experiments = experiments
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.results: List[ExperimentResult] = []

    def run_all(self):
        """Run all experiments sequentially."""
        print("=" * 80)
        print(f"EXPERIMENT RUNNER: {len(self.experiments)} experiments to run")
        print("=" * 80)

        for i, config in enumerate(self.experiments, 1):
            print(f"\n{'=' * 80}")
            print(f"EXPERIMENT {i}/{len(self.experiments)}: {config.name}")
            print(f"Description: {config.description}")
            print(f"Output: {config.get_output_dir()}")
            print(f"{'=' * 80}\n")

            result = self.run_experiment(config)
            self.results.append(result)

            if not result.success and not self.continue_on_error:
                print(f"\n[ERROR] Experiment {config.name} failed. Stopping.")
                break

        self._print_summary()
        self._save_all_results()

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration

        Returns:
            ExperimentResult with outcome and metrics
        """
        # Save configuration
        config.save()

        # Build command
        cmd = self._build_command(config)

        if self.dry_run:
            print("[DRY RUN] Would execute:")
            print(" ".join(cmd))
            print()
            return ExperimentResult(
                config=config,
                success=True,
                metrics={"dry_run": True},
            )

        # Execute training
        start_time = time.time()

        try:
            print("[RUNNING] Starting training...")
            print(f"Command: {' '.join(cmd)}\n")

            # Run training script
            subprocess.run(
                cmd,
                capture_output=False,  # Let output stream to console
                text=True,
                check=True,
            )

            end_time = time.time()

            # Load metrics from output
            metrics = self._load_metrics(config)

            result = ExperimentResult(
                config=config,
                success=True,
                metrics=metrics,
                start_time=start_time,
                end_time=end_time,
            )

            print(f"\n[SUCCESS] Experiment completed in {result.duration:.1f}s")
            print(f"Metrics: {metrics}")

        except subprocess.CalledProcessError as e:
            end_time = time.time()

            result = ExperimentResult(
                config=config,
                success=False,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
            )

            print(f"\n[FAILED] Experiment failed: {e}")

        except Exception as e:
            end_time = time.time()

            result = ExperimentResult(
                config=config,
                success=False,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
            )

            print(f"\n[ERROR] Unexpected error: {e}")

        # Save result
        result.save()

        return result

    def _build_command(self, config: ExperimentConfig) -> List[str]:
        """Build command to execute training."""
        # Use the current Python interpreter
        cmd = [sys.executable, "run.py"]
        cmd.extend(config.to_cli_args())
        return cmd

    def _load_metrics(self, config: ExperimentConfig) -> Dict:
        """Load evaluation metrics from trainer output."""
        metrics_file = os.path.join(
            config.get_trainer_output_dir(), "eval_metrics.json"
        )

        if not os.path.exists(metrics_file):
            print(f"[WARNING] Metrics file not found: {metrics_file}")
            return {}

        try:
            with open(metrics_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load metrics: {e}")
            return {}

    def _print_summary(self):
        """Print summary of all experiments."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful

        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        print("\nResults by experiment:")
        print("-" * 80)

        for result in self.results:
            status = "✓" if result.success else "✗"
            duration = f"{result.duration:.1f}s" if result.duration else "N/A"

            print(f"{status} {result.config.name:30s} | {duration:>10s} | ", end="")

            if result.success:
                # Print key metrics
                f1 = result.metrics.get("eval_f1", "N/A")
                exact_match = result.metrics.get("eval_exact_match", "N/A")
                print(f"F1={f1:.4f} EM={exact_match:.4f}")
            else:
                print(f"ERROR: {result.error}")

        print("=" * 80 + "\n")

    def _save_all_results(self):
        """Save aggregated results to a single file."""
        output_dir = (
            self.experiments[0].output_dir if self.experiments else "./experiments"
        )
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(output_dir, "all_results.json")

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "results": [r.to_dict() for r in self.results],
        }

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"[SAVED] All results saved to: {results_file}")


def run_experiments_from_configs(
    config_files: List[str],
    dry_run: bool = False,
    continue_on_error: bool = True,
):
    """
    Run experiments from configuration files.

    Args:
        config_files: List of paths to experiment config JSON files
        dry_run: If True, only print commands without executing
        continue_on_error: If True, continue running experiments even if one fails
    """
    experiments = []

    for config_file in config_files:
        try:
            config = ExperimentConfig.load(config_file)
            experiments.append(config)
        except Exception as e:
            print(f"[ERROR] Failed to load config {config_file}: {e}")

    if not experiments:
        print("[ERROR] No valid experiments to run")
        return

    runner = ExperimentRunner(
        experiments=experiments,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )

    runner.run_all()


def run_experiments_from_list(
    experiments: List[ExperimentConfig],
    dry_run: bool = False,
    continue_on_error: bool = True,
):
    """
    Run a list of experiments.

    Args:
        experiments: List of experiment configurations
        dry_run: If True, only print commands without executing
        continue_on_error: If True, continue running experiments even if one fails
    """
    runner = ExperimentRunner(
        experiments=experiments,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )

    runner.run_all()

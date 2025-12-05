#!/usr/bin/env python3
"""
Sweep launcher and manager for hyperparameter search experiments.

This script helps you launch and manage W&B sweeps for different experimental configurations:
1. Baseline runs (multiple seeds, no modifications)
2. Filtering strategies (cartography + cluster filtering)
3. Smoothing/Weighting strategies (label smoothing + soft weighting)
4. Combined strategies (filtering + smoothing/weighting)

Usage:
    # Launch a new sweep
    python sweep_launcher.py --sweep baseline --count 5
    python sweep_launcher.py --sweep filtering --count 20

    # Resume an existing sweep
    python sweep_launcher.py --resume <sweep_id> --count 10

    # List all sweeps in project
    python sweep_launcher.py --list

    # Get sweep status
    python sweep_launcher.py --status <sweep_id>
"""

import argparse
import os
import sys

import wandb


SWEEP_CONFIGS = {
    "baseline": "sweeps/baseline_sweep.yaml",
    "filtering": "sweeps/filtering_strategies_sweep.yaml",
    "smoothing": "sweeps/smoothing_weighting_sweep.yaml",
    "combined": "sweeps/combined_strategies_sweep.yaml",
}

PROJECT_NAME = "qa-cartography-experiments"


def launch_sweep(sweep_config_path: str, project: str = PROJECT_NAME) -> str:
    """
    Initialize a new W&B sweep.

    Args:
        sweep_config_path: Path to sweep YAML configuration
        project: W&B project name

    Returns:
        Sweep ID
    """
    import yaml

    if not os.path.exists(sweep_config_path):
        print(f"Error: Sweep config not found at {sweep_config_path}")
        sys.exit(1)

    print(f"Launching sweep from: {sweep_config_path}")

    # Load config as dict and initialize sweep
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    print(f"\n{'=' * 70}")
    print("Sweep initialized successfully!")
    print(f"{'=' * 70}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project}")
    print("\nTo run agents for this sweep:")
    print(f"  wandb agent {project}/{sweep_id}")
    print("\nOr use this script:")
    print(f"  python sweep_launcher.py --resume {sweep_id} --count <num_runs>")
    print(f"{'=' * 70}\n")

    return sweep_id


def run_sweep_agent(sweep_id: str, count: int = 1, project: str = PROJECT_NAME):
    """
    Run a W&B sweep agent.

    Args:
        sweep_id: W&B sweep ID (can be just the ID or entity/project/ID)
        count: Number of runs to execute
        project: W&B project name
    """
    # If sweep_id doesn't contain slashes, it's just the ID - need to add entity/project
    if "/" not in sweep_id:
        api = wandb.Api()
        entity = api.default_entity
        sweep_id = f"{entity}/{project}/{sweep_id}"

    print(f"Starting sweep agent for: {sweep_id}")
    print(f"Will execute {count} run(s)\n")

    # Run the agent
    wandb.agent(sweep_id=sweep_id, count=count)

    print(f"\n{'=' * 70}")
    print(f"Sweep agent completed {count} run(s)")
    print(f"{'=' * 70}\n")


def list_sweeps(project: str = PROJECT_NAME):
    """List all sweeps in the project."""
    api = wandb.Api()

    try:
        sweeps = api.project(project).sweeps()

        print(f"\n{'=' * 70}")
        print(f"Sweeps in project: {project}")
        print(f"{'=' * 70}\n")

        for sweep in sweeps:
            print(f"ID: {sweep.id}")
            print(f"  Name: {sweep.name}")
            print(f"  State: {sweep.state}")
            print(f"  Runs: {sweep.run_count}")
            print(f"  Created: {sweep.created_at}")

            # Best run info
            if sweep.best_run():
                best = sweep.best_run()
                print(f"  Best run: {best.name}")
                if hasattr(best, "summary"):
                    f1 = best.summary.get("eval/f1", "N/A")
                    print(f"    F1: {f1}")
            print()

    except Exception as e:
        print(f"Error listing sweeps: {e}")
        print("Make sure you're logged in: wandb login")


def get_sweep_status(sweep_id: str, project: str = PROJECT_NAME):
    """Get detailed status of a sweep."""
    api = wandb.Api()

    try:
        sweep = api.sweep(f"{project}/{sweep_id}")

        print(f"\n{'=' * 70}")
        print(f"Sweep Status: {sweep.id}")
        print(f"{'=' * 70}\n")

        print(f"Name: {sweep.name}")
        print(f"State: {sweep.state}")
        print(f"Method: {sweep.config.get('method', 'N/A')}")
        print(f"Total runs: {sweep.run_count}")
        print(f"Created: {sweep.created_at}")
        print(f"Updated: {sweep.updated_at}")

        # Configuration
        print("\nMetric to optimize:")
        metric = sweep.config.get("metric", {})
        print(f"  {metric.get('name', 'N/A')} ({metric.get('goal', 'N/A')})")

        # Best run
        print("\nBest run:")
        if sweep.best_run():
            best = sweep.best_run()
            print(f"  Name: {best.name}")
            print(f"  ID: {best.id}")
            if hasattr(best, "summary"):
                print("  Metrics:")
                for key in ["eval/f1", "eval/exact_match", "train/loss"]:
                    if key in best.summary:
                        print(f"    {key}: {best.summary[key]:.4f}")
        else:
            print("  No completed runs yet")

        # Recent runs
        print("\nRecent runs:")
        runs = list(sweep.runs)[:5]  # Last 5 runs
        for run in runs:
            status = run.state
            print(f"  {run.name} - {status}")
            if status == "finished" and hasattr(run, "summary"):
                f1 = run.summary.get("eval/f1", "N/A")
                print(f"    F1: {f1}")

        print(f"\n{'=' * 70}\n")

    except Exception as e:
        print(f"Error getting sweep status: {e}")
        print("Make sure the sweep ID is correct and you're logged in")


def main():
    parser = argparse.ArgumentParser(
        description="Launch and manage W&B hyperparameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch baseline sweep with 5 runs
  python sweep_launcher.py --sweep baseline --count 5
  
  # Launch filtering strategies sweep
  python sweep_launcher.py --sweep filtering --count 20
  
  # Resume an existing sweep
  python sweep_launcher.py --resume abc123 --count 10
  
  # List all sweeps
  python sweep_launcher.py --list
  
  # Check sweep status
  python sweep_launcher.py --status abc123
        """,
    )

    parser.add_argument(
        "--sweep",
        choices=list(SWEEP_CONFIGS.keys()),
        help="Type of sweep to launch (baseline, filtering, smoothing, combined)",
    )

    parser.add_argument(
        "--resume", type=str, metavar="SWEEP_ID", help="Resume an existing sweep by ID"
    )

    parser.add_argument(
        "--count", type=int, default=1, help="Number of runs to execute (default: 1)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default=PROJECT_NAME,
        help=f"W&B project name (default: {PROJECT_NAME})",
    )

    parser.add_argument(
        "--list", action="store_true", help="List all sweeps in the project"
    )

    parser.add_argument(
        "--status", type=str, metavar="SWEEP_ID", help="Get detailed status of a sweep"
    )

    args = parser.parse_args()

    # Handle different commands
    if args.list:
        list_sweeps(args.project)
        return

    if args.status:
        get_sweep_status(args.status, args.project)
        return

    if args.resume:
        # Resume existing sweep
        run_sweep_agent(args.resume, args.count, args.project)
        return

    if args.sweep:
        # Launch new sweep
        config_path = SWEEP_CONFIGS[args.sweep]
        sweep_id = launch_sweep(config_path, args.project)

        # Ask if user wants to start agent now
        response = input(f"\nStart agent now with {args.count} run(s)? [y/N]: ")
        if response.lower() in ["y", "yes"]:
            run_sweep_agent(sweep_id, args.count, args.project)
        return

    # No command specified
    parser.print_help()


if __name__ == "__main__":
    main()

# Experiment System - Quick Reference

## Installation & Setup

No additional installation needed - uses existing project dependencies.

## Before Running Experiments

### Generate Cartography Metrics (Required for most experiments)

```bash
# For training set
python run.py \
    --do_train \
    --enable_cartography \
    --train_split train \
    --cartography_output_dir ./cartography_output \
    --max_train_samples 10000 \
    --num_train_epochs 5 \
    --output_dir ./cartography_train

# For validation set (if using filtering with --filter_validation)
python run.py \
    --do_train \
    --enable_cartography \
    --train_split validation \
    --cartography_output_dir ./cartography_output_validation \
    --max_train_samples 2000 \
    --num_train_epochs 5 \
    --output_dir ./cartography_val
```

### Generate Cluster Assignments (Optional - for cluster filtering)

```bash
python extract_embeddings.py --max_samples 10000
python cluster_analysis.py
```

## Running Experiments

### Quick Test (2 experiments, small dataset)

```bash
python run_experiments.py --mode run --suite minimal \
    --max-train-samples 1000 --max-eval-samples 500
```

### Sample Suite (~10 experiments)

```bash
python run_experiments.py --mode run --suite sample \
    --max-train-samples 10000
```

### Filtering Suite (3 experiments: baseline + filtering methods)

```bash
python run_experiments.py --mode run --suite filtering \
    --max-train-samples 10000 \
    --cartography-dir ./cartography_output \
    --validation-cartography-dir ./cartography_output_validation \
    --cluster-dir ./cluster_output
```

**Note**: This suite filters BOTH training and validation sets.

### Weighting Suite (3 experiments: baseline + weighting methods)

```bash
python run_experiments.py --mode run --suite weighting \
    --max-train-samples 10000 \
    --cartography-dir ./cartography_output
```

### With Multiple Seeds (for statistical significance)

```bash
python run_experiments.py --mode run --suite sample \
    --max-train-samples 10000 \
    --seeds 42 43 44
```

### Dry Run (preview without executing)

```bash
python run_experiments.py --mode run --suite sample --dry-run
```

### From Config Files

```bash
python run_experiments.py --mode run \
    --configs example_configs/*.json
```

## Analyzing Results

### Full Analysis (summary + plots + CSV export)

```bash
python run_experiments.py --mode analyze
```

### With Interactive Plots

```bash
python run_experiments.py --mode analyze --show-plots
```

## Available Experiment Types

| Type                   | Description                       | Key Parameter                   |
| ---------------------- | --------------------------------- | ------------------------------- |
| **Baseline**           | Standard training                 | None                            |
| **Cartography Filter** | Drop ambiguous samples            | `top_fraction` (e.g., 0.33)     |
| **Cluster Filter**     | Exclude specific clusters         | `exclude_clusters` (e.g., [-1]) |
| **Label Smoothing**    | Soft targets based on variability | `smoothing_factor` (e.g., 0.6)  |
| **Soft Weighting**     | Weight loss by variability        | `weight_clip_min/max`           |

## Common Options

| Option                         | Description                    | Default                         |
| ------------------------------ | ------------------------------ | ------------------------------- |
| `--max-train-samples`          | Limit training samples         | 10000                           |
| `--max-eval-samples`           | Limit eval samples             | 2000                            |
| `--seeds`                      | Random seeds for replication   | 42                              |
| `--cartography-dir`            | Training cartography metrics   | ./cartography_output            |
| `--validation-cartography-dir` | Validation cartography metrics | ./cartography_output_validation |
| `--cluster-dir`                | Cluster assignments directory  | ./cluster_output                |
| `--dry-run`                    | Preview without executing      | False                           |
| `--stop-on-error`              | Stop on first failure          | False (continues)               |

## Output Structure

```
experiments/
├── all_results.json          # Aggregated results
├── results.csv               # Results in CSV
├── experiment_results.png    # Plots
└── <experiment_name>_<hash>/
    ├── config.json           # Configuration
    ├── results.json          # Results
    └── trainer_output/       # Model & metrics
```

## Key Metrics Tracked

- `eval_f1`: F1 score
- `eval_exact_match`: Exact match accuracy
- `duration_seconds`: Training time
- `success`: Completion status

## Common Issues & Solutions

| Issue                           | Solution                                     |
| ------------------------------- | -------------------------------------------- |
| "Cartography metrics not found" | Run cartography first (see above)            |
| "Cluster assignments not found" | Run clustering or disable cluster filtering  |
| Out of memory                   | Reduce batch size or sample count            |
| Too slow                        | Use `--suite minimal` or reduce sample count |

## Example Workflows

### 1. Quick Validation

```bash
# Test with small dataset
python run_experiments.py --mode run --suite minimal \
    --max-train-samples 1000 --max-eval-samples 500 --dry-run

# If OK, run for real
python run_experiments.py --mode run --suite minimal \
    --max-train-samples 1000 --max-eval-samples 500
```

### 2. Full Experiment Suite

```bash
# 1. Generate cartography for training
python run.py --do_train --enable_cartography \
    --train_split train \
    --cartography_output_dir ./cartography_output \
    --max_train_samples 10000 --num_train_epochs 5

# 1b. Generate cartography for validation (optional)
python run.py --do_train --enable_cartography \
    --train_split validation \
    --cartography_output_dir ./cartography_output_validation \
    --max_train_samples 2000 --num_train_epochs 5

# 2. Run experiments
python run_experiments.py --mode run --suite filtering \
    --max-train-samples 10000 --seeds 42 43 44

# 3. Analyze
python run_experiments.py --mode analyze --show-plots
```

### 3. Custom Experiments

```python
# Create custom_experiments.py
from experiment_config import *
from experiment_runner import run_experiments_from_list

experiments = [
    create_baseline_config(name="my_baseline", seed=42),
    create_cartography_filter_config(name="my_filter", top_fraction=0.25),
]

run_experiments_from_list(experiments)
```

Then run:

```bash
python custom_experiments.py
```

## Python API Quick Reference

### Creating Experiments

```python
from experiment_config import (
    create_baseline_config,
    create_cartography_filter_config,
    create_label_smoothing_config,
)

# Baseline
exp1 = create_baseline_config(
    name="baseline",
    seed=42,
    max_train_samples=10000,
)

# Cartography filtering
exp2 = create_cartography_filter_config(
    name="cart_filter",
    top_fraction=0.33,
    cartography_dir="./cartography_output",
)

# Label smoothing
exp3 = create_label_smoothing_config(
    name="smooth",
    smoothing_factor=0.6,
)
```

### Running Experiments

```python
from experiment_runner import run_experiments_from_list

run_experiments_from_list(
    experiments=[exp1, exp2, exp3],
    dry_run=False,
    continue_on_error=True,
)
```

### Analyzing Results

```python
from experiment_analysis import (
    load_experiment_results,
    print_results_summary,
    plot_results,
)

df = load_experiment_results("./experiments")
print_results_summary(df)
plot_results(df, show_plots=True)
```

## Tips

1. **Start small**: Use `--suite minimal` with small sample counts for testing
2. **Use seeds**: Run with multiple seeds (e.g., `--seeds 42 43 44`) for statistical significance
3. **Monitor progress**: Watch console output for training progress and errors
4. **Dry run first**: Use `--dry-run` to preview experiments before running
5. **Check disk space**: Each experiment saves model checkpoints (~100MB each)

## Documentation

- Full guide: `EXPERIMENT_GUIDE.md`
- Example configs: `example_configs/`
- Code documentation: Inline docstrings in Python files

## Support Files

- `experiment_config.py` - Configuration system
- `experiment_runner.py` - Execution engine
- `experiment_analysis.py` - Analysis & visualization
- `run_experiments.py` - Main CLI interface

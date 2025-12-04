# Experiment System Guide

This guide explains how to use the experiment system to run systematic training experiments with different configurations and analyze results.

## Overview

The experiment system provides:

1. **Structured Configuration**: Define experiments with different filtering and training strategies
2. **Automated Execution**: Run multiple experiments sequentially with automatic tracking
3. **Result Analysis**: Analyze and visualize results across experiments
4. **Reproducibility**: Built-in seeding and configuration saving for reproducible results

## Quick Start

### 1. Prerequisites

Before running experiments, you need to have:

- Cartography metrics from a previous training run (for filtering/smoothing/weighting strategies)
- Optionally: Cluster assignments (for cluster-based filtering)
- Optionally: If you're using filtering strategy you need to run cartography on validation split as well.
- Make sure you keep track of where you are saving the results of all the above otherwise they might be overwritten.

Generate cartography metrics:

```bash
python run.py \
    --do_train --do_eval \
    --enable_cartography \
    --cartography_output_dir ./cartography_output \
    --max_train_samples 10000 \
    --num_train_epochs 5 \
    --output_dir ./initial_training
```

### 2. Run Experiments

Run a minimal experiment suite (2 experiments):

```bash
python run_experiments.py --mode run --suite minimal --max-train-samples 5000
```

Run a full sample suite (~10 experiments per seed):

```bash
python run_experiments.py --mode run --suite sample --max-train-samples 10000
```

Run with multiple seeds for statistical significance:

```bash
python run_experiments.py --mode run --suite sample \
    --max-train-samples 10000 \
    --seeds 42 43 44
```

Preview experiments without running (dry-run):

```bash
python run_experiments.py --mode run --suite sample --dry-run
```

### 3. Analyze Results

After experiments complete, analyze the results:

```bash
python run_experiments.py --mode analyze
```

This will:

- Load all experiment results
- Print summary statistics
- Generate comparison plots
- Export results to CSV

## Experiment Types

### 1. Baseline

Standard training without any filtering or special strategies.

### 2. Cartography Filtering

Filters training data based on dataset cartography:

- Removes ambiguous examples (high variability, low confidence)
- Keeps top N% most ambiguous + all easy/hard examples
- Requires pre-computed cartography metrics

**Parameters:**

- `top_fraction`: Fraction of most ambiguous examples to keep (e.g., 0.33 = 33%)

### 3. Cluster Filtering

Filters training data based on clustering:

- Excludes specific clusters (e.g., noise cluster)
- Optionally filters by cluster probability threshold
- Requires pre-computed cluster assignments

**Parameters:**

- `exclude_clusters`: List of cluster IDs to exclude (e.g., [-1] for noise)
- `min_cluster_probability`: Minimum probability threshold

### 4. Label Smoothing

Applies variability-based label smoothing:

- Higher variability → more smoothing
- Reduces overconfidence on ambiguous examples
- Requires pre-computed cartography metrics

**Parameters:**

- `smoothing_factor`: Multiplier for smoothing (default: 0.6)

### 5. Soft Weighting

Applies loss weighting based on variability:

- Higher variability → higher weight
- Focuses learning on harder examples
- Requires pre-computed cartography metrics

**Parameters:**

- `weight_clip_min`: Minimum weight (default: 0.1)
- `weight_clip_max`: Maximum weight (default: 10.0)

## Custom Experiments

### Creating Custom Configurations

```python
from experiment_config import (
    ExperimentConfig,
    TrainingConfig,
    CartographyFilterConfig,
)

# Create custom training config
training = TrainingConfig(
    seed=42,
    max_train_samples=5000,
    num_train_epochs=3.0,
    learning_rate=5e-5,
    cartography_filter=CartographyFilterConfig(
        enabled=True,
        cartography_output_dir="./cartography_output",
        top_fraction=0.25,
    ),
)

# Create experiment
experiment = ExperimentConfig(
    name="custom_experiment",
    description="My custom experiment with 25% ambiguous filtering",
    training=training,
)

# Save configuration
experiment.save()
```

### Running Custom Experiments

```python
from experiment_runner import run_experiments_from_list

experiments = [
    # Add your custom experiments here
]

run_experiments_from_list(experiments=experiments)
```

## Advanced Usage

### Combining Multiple Strategies

You can combine filtering with training strategies:

```python
from experiment_config import (
    ExperimentConfig,
    TrainingConfig,
    CartographyFilterConfig,
    LabelSmoothingConfig,
)

training = TrainingConfig(
    seed=42,
    max_train_samples=5000,
    # Filter ambiguous examples
    cartography_filter=CartographyFilterConfig(
        enabled=True,
        cartography_output_dir="./cartography_output",
        top_fraction=0.33,
    ),
    # Apply label smoothing
    label_smoothing=LabelSmoothingConfig(
        enabled=True,
        cartography_output_dir="./cartography_output",
        smoothing_factor=0.6,
    ),
)

experiment = ExperimentConfig(
    name="combined_strategies",
    description="Cartography filtering + label smoothing",
    training=training,
)
```

### Running Experiments from Config Files

Save experiment configs and run later:

```bash
# Experiments are auto-saved when run
# Find configs in: experiments/*/config.json

# Re-run specific experiments
python run_experiments.py --mode run \
    --configs experiments/baseline_*/config.json experiments/cartography_*/config.json
```

### Analyzing Specific Experiments

```python
from experiment_analysis import (
    load_experiment_results,
    compare_experiments,
    print_results_summary,
)

# Load all results
df = load_experiment_results("./experiments")

# Compare specific experiments
compare_experiments(
    df,
    experiment_names=["baseline", "cartography_filter", "label_smoothing"],
    metrics=["eval_f1", "eval_exact_match"],
)

# Print summary
print_results_summary(df)
```

## Output Structure

```
experiments/
├── all_results.json              # Aggregated results from all experiments
├── results.csv                    # Results in CSV format
├── experiment_results.png         # Visualization plots
├── baseline_abc12345/             # Individual experiment directory
│   ├── config.json                # Experiment configuration
│   ├── results.json               # Experiment results
│   └── trainer_output/            # HuggingFace trainer output
│       ├── eval_metrics.json      # Evaluation metrics
│       ├── eval_predictions.jsonl # Predictions
│       ├── config.json            # Model config
│       ├── model.safetensors      # Trained model
│       └── ...
└── cartography_filter_def67890/
    └── ...
```

## Result Metrics

The system tracks the following metrics:

- **eval_f1**: F1 score on evaluation set
- **eval_exact_match**: Exact match score
- **duration_seconds**: Training time
- **success**: Whether experiment completed successfully

## Tips for Running Experiments

### 1. Start Small

Begin with a minimal suite to verify everything works:

```bash
python run_experiments.py --mode run --suite minimal \
    --max-train-samples 1000 --max-eval-samples 500
```

### 2. Use Multiple Seeds

Run with multiple seeds for statistical significance:

```bash
python run_experiments.py --mode run --suite sample --seeds 42 43 44
```

### 3. Monitor Progress

Experiments run sequentially and print progress. Check the console output to monitor:

- Which experiment is currently running
- Training progress (loss, metrics)
- Success/failure status

### 4. Handle Failures

By default, the system continues on errors. To stop on first failure:

```bash
python run_experiments.py --mode run --suite sample --stop-on-error
```

### 5. Resource Management

Each experiment saves checkpoints. To limit disk usage:

- Use smaller `max_train_samples` for testing
- Set `save_total_limit` in TrainingConfig
- Clean up checkpoint directories after experiments

### 6. Reproducibility

All experiments include:

- Fixed random seed
- Saved configuration
- Git commit hash (if in a git repo)
- Timestamp

## Troubleshooting

### Issue: "Cartography metrics not found"

**Solution**: Run cartography first on your dataset:

```bash
python run.py --do_train --do_eval --enable_cartography \
    --max_train_samples 10000 --output_dir ./initial_training
```

### Issue: "Cluster assignments not found"

**Solution**: Run clustering first:

```bash
python extract_embeddings.py --max_samples 10000
python cluster_analysis.py
```

### Issue: "Out of memory"

**Solution**: Reduce batch size or sample count:

```python
training = TrainingConfig(
    per_device_train_batch_size=4,  # Reduce from default 8
    max_train_samples=5000,
)
```

### Issue: Experiments running too slowly

**Solution**:

- Reduce `max_train_samples` and `max_eval_samples`
- Reduce `num_train_epochs`
- Use fewer experiments in your suite
- Use `--suite minimal` for quick testing

## Example Workflows

### Workflow 1: Testing Different Filtering Strategies

```bash
# 1. Generate cartography metrics
python run.py --do_train --do_eval --enable_cartography \
    --max_train_samples 10000 --num_train_epochs 3

# 2. Run experiments with different filtering
python run_experiments.py --mode run --suite sample \
    --max-train-samples 10000

# 3. Analyze results
python run_experiments.py --mode analyze --show-plots
```

### Workflow 2: Statistical Significance Testing

```bash
# Run same experiments with multiple seeds
python run_experiments.py --mode run --suite sample \
    --max-train-samples 10000 \
    --seeds 42 43 44 45 46

# Analyze with statistical comparison
python run_experiments.py --mode analyze
```

### Workflow 3: Quick Validation

```bash
# Quick test with small dataset
python run_experiments.py --mode run --suite minimal \
    --max-train-samples 1000 --max-eval-samples 500 \
    --dry-run  # Preview first

# Run if looks good
python run_experiments.py --mode run --suite minimal \
    --max-train-samples 1000 --max-eval-samples 500
```

## Next Steps

- Customize experiment configurations for your needs
- Add new experiment types by extending the config system
- Integrate with your existing training pipeline
- Export results for publication-ready figures

# Training Script Quick Reference

## Quick Start

```bash
./run_training.sh
```

The script will ask you:

1. **Configuration source**: Choose between custom config or W&B config
2. **Run name**: Identifier for this training run (e.g., "baseline", "filtered_v2")
3. **Max training samples**: 0 = full dataset (~130K), or specify a number (e.g., 5000)
4. **Max eval samples**: 0 = full dataset (~33K), or specify a number (e.g., 1000)

## Configuration Files

### Option 1: Custom Configuration (`train_config_custom.yaml`)

Your own training configuration with custom parameters.

**Location**: `train_config_custom.yaml`

Edit this file to set:

- Model and dataset
- Training hyperparameters
- Filtering strategies
- W&B settings

**Note**: Dataset sizes are specified interactively, not in the config file.

### Option 2: W&B Configuration (`train_config_wandb.yaml`)

Use configuration from a successful W&B sweep run.

**Location**: `train_config_wandb.yaml`

To create this:

1. Find your best W&B run in the web UI
2. Download the `config.yaml` from that run (use the Files tab or W&B API)
3. Save it as `train_config_wandb.yaml`
4. The script will automatically parse the W&B format and extract parameter values

**Note**: The W&B config format has each parameter under a `value:` key. The script handles this automatically.

## Output Directories

```
./trained_models/train_{RUN_NAME}/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── eval_metrics.json        # Evaluation metrics
└── eval_predictions.jsonl   # Predictions

./evaluations/{RUN_NAME}/
├── training.log             # Full training log
├── eval_metrics.json        # Copy of evaluation results
└── eval_predictions.jsonl   # Copy of predictions
```

## Example Workflow

### Scenario 1: Quick Test with Custom Config

```bash
./run_training.sh
# Choose: 1 (custom config)
# Run name: quick_test
# Max train: 1000
# Max eval: 500
```

### Scenario 2: Use Best W&B Sweep Config

```bash
# First, download your best W&B config from the web UI (Files tab)
# Save it as train_config_wandb.yaml

./run_training.sh
# Choose: 2 (W&B config)
# Run name: wandb_best
# Max train: 0 (full dataset)
# Max eval: 0 (full dataset)
```

### Scenario 3: Full Training with Custom Settings

```bash
# Edit train_config_custom.yaml with your parameters
nano train_config_custom.yaml

./run_training.sh
# Choose: 1 (custom config)
# Run name: final_model
# Max train: 0 (full dataset)
# Max eval: 0 (full dataset)
```

## Tips

- **For experiments**: Use 5K train / 1K eval samples for faster iteration
- **For final models**: Use 0 for both (full dataset)
- **Run naming**: Use descriptive names like "cluster_filtered_v2" or "smoothed_baseline"
- **W&B configs**: Great for reproducing best sweep results on full dataset
- **Custom configs**: Better for trying new combinations not explored in sweeps

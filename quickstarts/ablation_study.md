# Ablation Study Script Quick Reference

## Quick Start

```bash
./run_ablation_study.sh
```

The script will ask you:

1. **Max training samples**: 0 = full dataset (~130K), or specify a number (e.g., 5000)
2. **Max eval samples**: 0 = full dataset (~33K), or specify a number (e.g., 1000)
3. **Number of seeds**: How many random seeds to run (default: 5)
4. **Batch size**: Training batch size (default: 64)
5. **Skip completed runs**: Whether to resume from previous runs (default: yes)

## What It Does

The ablation study compares three model variants across multiple random seeds:

- **none** (baseline): Full model with question + passage
- **q_only**: Question-only (passage is masked out)
- **p_only**: Passage-only (question replaced with generic prompt)

For each variant, it runs training with multiple seeds and generates:

- Statistical comparison (mean, std, confidence intervals)
- Statistical tests (t-tests, effect sizes)
- Visualizations (box plots, line plots, bar charts)

## Output Structure

```
./experiments/
├── ablation_none_seed_42/
│   ├── eval_metrics.json
│   └── eval_predictions.jsonl
├── ablation_q_only_seed_42/
│   └── ...
├── ablation_p_only_seed_42/
│   └── ...
├── ablation_analysis/
│   ├── ablation_comparison.csv      # Summary statistics
│   ├── ablation_results.csv         # All raw results
│   ├── ablation_boxplot.png         # Box plot comparison
│   ├── ablation_barplot.png         # Bar plot with error bars
│   └── ablation_lineplot.png        # Seed-by-seed comparison
└── ablation_*.log                    # Training logs for each run
```

## Example Workflows

### Scenario 1: Quick Test (Small Dataset)

```bash
./run_ablation_study.sh

# Max training samples: 1000
# Max eval samples: 500
# Number of seeds: 3
# Batch size: 64
# Skip completed runs: yes
```

**Time**: ~30-45 minutes for 9 runs (3 ablations × 3 seeds)

### Scenario 2: Standard Experiment (Medium Dataset)

```bash
./run_ablation_study.sh

# Max training samples: 5000
# Max eval samples: 1000
# Number of seeds: 5
# Batch size: 64
# Skip completed runs: yes
```

**Time**: ~2-3 hours for 15 runs (3 ablations × 5 seeds)

### Scenario 3: Full Dataset Experiment

```bash
./run_ablation_study.sh

# Max training samples: 0  (full ~130K)
# Max eval samples: 0       (full ~33K)
# Number of seeds: 5
# Batch size: 64
# Skip completed runs: yes
```

**Time**: ~10-15 hours for 15 runs (3 ablations × 5 seeds)

### Scenario 4: Resume Interrupted Study

```bash
./run_ablation_study.sh

# (Use same parameters as before)
# Skip completed runs: yes  ← This will skip already completed runs
```

## Analysis Output

After completion, the script generates `./experiments/ablation_analysis/ablation_comparison.csv`:

```
ablation,f1_mean,f1_std,f1_ci_lower,f1_ci_upper,em_mean,em_std,em_ci_lower,em_ci_upper,num_seeds
none,0.7234,0.0123,0.7089,0.7379,0.6123,0.0156,0.5934,0.6312,5
q_only,0.6521,0.0145,0.6354,0.6688,0.5234,0.0167,0.5038,0.5430,5
p_only,0.5123,0.0198,0.4893,0.5353,0.3921,0.0212,0.3672,0.4170,5
```

## Statistical Tests

The analysis includes:

- **Paired t-tests**: Compare each ablation to baseline
- **Effect sizes**: Cohen's d for practical significance
- **Confidence intervals**: 95% CI for mean metrics

Results are printed to console and saved in logs.

## Visualizations

Three plots are generated automatically:

1. **Box Plot** (`ablation_boxplot.png`): Distribution of results across seeds
2. **Bar Plot** (`ablation_barplot.png`): Mean performance with error bars (95% CI)
3. **Line Plot** (`ablation_lineplot.png`): Seed-by-seed comparison

## Configuration

Default settings (can be changed interactively):

- Model: `google/electra-small-discriminator`
- Dataset: `Eladio/emrqa-msquad`
- Epochs: 3
- Max length: 512
- W&B Project: `qa-ablation-study`

## Tips

### For Quick Testing

- Use 1000-2000 train samples
- Use 3 seeds minimum
- Results in ~30 min

### For Publication/Final Results

- Use full dataset (0 for both)
- Use 5+ seeds for statistical power
- Allow 10-15 hours
- Keep all training logs

### For Debugging

- Use very small dataset (100-500 samples)
- Use 1-2 seeds
- Results in ~5-10 min

### Resuming Work

- Always answer "yes" to skip completed runs
- The script detects existing `eval_metrics.json` files
- You can add more seeds by running again with higher count

## Cleanup

At the end, the script offers to remove model checkpoints:

- Answer "yes" to keep only metrics and predictions
- This can save significant disk space
- Model weights are removed, metrics are kept

## Troubleshooting

### "Out of memory" errors

- Reduce `--batch_size` (try 32 or 16)
- Reduce `--max_train_samples`

### Script takes too long

- Reduce number of seeds
- Use smaller dataset
- Consider using GPU if available

### Results look strange

- Check training logs in `./experiments/ablation_*.log`
- Verify dataset loaded correctly
- Try with more seeds for stability

## Comparison with Other Scripts

| Script                  | Purpose                           | Best For                                   |
| ----------------------- | --------------------------------- | ------------------------------------------ |
| `run_ablation_study.sh` | Compare ablations with statistics | Understanding what model components matter |
| `run_training.sh`       | Single config-based run           | Controlled experiments, final training     |
| `sweep_launcher.py`     | Hyperparameter search             | Finding optimal configurations             |

## Next Steps

After running ablations:

1. Review visualizations in `./experiments/ablation_analysis/`
2. Check statistical significance in console output
3. Use insights to decide which components to include
4. Run full training with best configuration using `run_training.sh`

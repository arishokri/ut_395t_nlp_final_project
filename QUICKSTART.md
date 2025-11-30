# Dataset Cartography - Quick Start Guide

## What was implemented?

I've implemented **Dataset Cartography** (Swayamdipta et al. 2020) for your EMR-QA project. This helps you identify which training examples are:

- **Easy to learn** (high confidence, low variability)
- **Hard to learn** (low confidence, low variability)
- **Ambiguous** (low confidence, high variability - potential label noise)

## Files Created

1. **`dataset_cartography.py`** - Core implementation with callback and analysis utilities
2. **`analyze_cartography.py`** - Standalone analysis script
3. **`cartography_example.ipynb`** - Interactive notebook with examples
4. **`CARTOGRAPHY_README.md`** - Comprehensive documentation
5. **`QUICKSTART.md`** - This file

## Files Modified

1. **`helpers.py`** - Added cartography tracking to `QuestionAnsweringTrainer`
2. **`run.py`** - Added `--enable_cartography` flag and callback integration

## Quick Start (3 Steps)

### Step 1: Train with Cartography

```bash
python run.py \
  --do_train \
  --dataset Eladio/emrqa-msquad \
  --model google/electra-small-discriminator \
  --output_dir ./trained_models/qa/test_cartography \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --max_train_samples 5000 \
  --enable_cartography \
  --cartography_output_dir ./cartography_output
```

**Key points:**

- Use `--enable_cartography` to track training dynamics
- Use at least 3-5 epochs for meaningful statistics
- Start with a subset (e.g., 5000 examples) for testing

### Step 2: Analyze Results

```bash
python analyze_cartography.py \
  --cartography_dir ./cartography_output \
  --dataset Eladio/emrqa-msquad \
  --n_examples 20
```

This generates:

- Cartography map visualization
- Category distribution plots
- Question type analysis
- Example samples from each category

### Step 3: Explore Interactively

Open the Jupyter notebook:

```bash
jupyter notebook cartography_example.ipynb
```

This lets you:

- Visualize the data map
- Examine specific hard/ambiguous examples
- Combine with your error analysis
- Create filtered datasets

## What You Get

### Metrics for Each Training Example

- **Confidence**: Average probability the model assigns to correct answer (0-1)
- **Variability**: How much the model's confidence fluctuates (0-~0.5)
- **Correctness**: Fraction of epochs the model predicted correctly (0-1)

### Example Categories

| Category      | Confidence | Variability | Interpretation                            |
| ------------- | ---------- | ----------- | ----------------------------------------- |
| **Easy**      | High       | Low         | Model learns quickly and consistently     |
| **Hard**      | Low        | Low         | Model struggles but is consistent         |
| **Ambiguous** | Low        | High        | Model can't decide - possible label noise |

### Output Files

After training with cartography:

```
cartography_output/
├── training_dynamics_epoch_1.json
├── training_dynamics_epoch_2.json
├── ...
├── cartography_metrics.json
├── cartography_metrics.csv
└── cartography_map.png
```

After analysis:

```
cartography_analysis/
├── question_type_analysis.csv
├── category_samples.json
├── metric_distributions.png
├── confidence_vs_correctness.png
└── category_analysis.png
```

## Use Cases

### 1. Identify Label Noise

```python
from dataset_cartography import load_cartography_metrics, categorize_examples

df = load_cartography_metrics("./cartography_output")
df = categorize_examples(df)

# Get ambiguous examples - candidates for relabeling
ambiguous = df[df['category'] == 'ambiguous']
print(f"Found {len(ambiguous)} potentially mislabeled examples")
```

### 2. Focus Training on Hard Examples

```python
# Get IDs of hard examples
hard_ids = df[df['category'] == 'hard'].index.tolist()

# Use these IDs to filter your training data
# Then retrain with focused curriculum
```

### 3. Analyze Error Patterns

```python
# Combine with your eval results
import pandas as pd

eval_df = pd.read_json("eval_baseline_emrqa/eval_predictions.jsonl", lines=True)
combined = eval_df.merge(df, left_on='id', right_index=True)

# See which categories have higher error rates
error_by_category = combined.groupby('category').apply(
    lambda x: (x['predicted_answer'] != x['answers'].apply(lambda a: a['text'][0])).mean()
)
print(error_by_category)
```

## Integration with Your Existing Analysis

You can combine cartography with your existing analysis in `data_analysis.ipynb`:

```python
# Your existing failed examples analysis
failed = pd.DataFrame(failed_examples)

# Add cartography metrics
cartography_df = load_cartography_metrics("./cartography_output")
failed_with_cart = failed.merge(
    cartography_df,
    left_on='id',
    right_index=True
)

# Now analyze failed examples by cartography category
print(failed_with_cart.groupby('category').size())
```

## Tips

1. **Start small**: Test on 5k-10k examples first
2. **Train longer**: Use 5+ epochs for stable estimates
3. **Compare models**: See how different architectures handle same examples
4. **Iterate**: Use insights to improve data → retrain → compare cartography maps

## Next Steps

1. ✅ Run a test training with `--enable_cartography` on a subset
2. ✅ Examine the generated cartography map
3. ✅ Look at hard and ambiguous examples
4. ✅ Decide on data cleaning/augmentation strategy
5. ✅ Retrain and compare results

## Questions?

See `CARTOGRAPHY_README.md` for detailed documentation including:

- Detailed API reference
- Troubleshooting guide
- Advanced usage examples
- Performance considerations

## Reference

Swayamdipta, S., et al. (2020). **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics**. EMNLP 2020. [arXiv:2009.10795](https://arxiv.org/abs/2009.10795)

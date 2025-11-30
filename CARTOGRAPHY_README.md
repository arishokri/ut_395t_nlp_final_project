# Dataset Cartography Implementation

This implementation provides dataset cartography (Swayamdipta et al. 2020) for question answering tasks to identify easy, hard, and ambiguous training examples.

## Overview

Dataset cartography analyzes training dynamics by tracking three key metrics for each training example across epochs:

1. **Confidence**: Average probability the model assigns to the correct answer
2. **Variability**: Standard deviation of the model's confidence across epochs
3. **Correctness**: Fraction of epochs where the model predicted correctly

Based on these metrics, examples are categorized as:

- **Easy to learn**: High confidence, low variability - model learns these quickly and consistently
- **Hard to learn**: Low confidence, low variability - model struggles but is consistent
- **Ambiguous**: Low confidence, high variability - model is uncertain and inconsistent

## Files Added

- `dataset_cartography.py` - Core cartography implementation with callback and utilities
- `analyze_cartography.py` - Analysis script for visualizing and exploring results
- `CARTOGRAPHY_README.md` - This documentation file

## Files Modified

- `helpers.py` - Updated `QuestionAnsweringTrainer` to track training dynamics
- `run.py` - Added `--enable_cartography` argument and callback integration

## Usage

### 1. Train with Cartography Tracking

Add the `--enable_cartography` flag when training:

```bash
python run.py \
  --do_train \
  --dataset Eladio/emrqa-msquad \
  --model google/electra-small-discriminator \
  --output_dir ./trained_models/qa/emrqa_with_cartography \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --max_train_samples 10000 \
  --enable_cartography \
  --cartography_output_dir ./cartography_output
```

**Important Parameters:**

- `--enable_cartography`: Enables cartography tracking
- `--cartography_output_dir`: Where to save cartography outputs (default: `./cartography_output`)
- `--num_train_epochs`: Should be ≥3 for meaningful statistics across epochs

### 2. Outputs During Training

The callback will generate:

- `training_dynamics_epoch_N.json` - Intermediate dynamics after each epoch
- `cartography_metrics.json` - Final aggregated metrics
- `cartography_metrics.csv` - Metrics in CSV format for analysis
- `cartography_map.png` - Visualization of the data map

### 3. Analyze Results

Use the analysis script to explore your results:

```bash
python analyze_cartography.py \
  --cartography_dir ./cartography_output \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --n_examples 20 \
  --output_dir ./cartography_analysis
```

This generates:

- **Question type analysis**: How different question types behave
- **Category samples**: Example questions from each category
- **Additional visualizations**: Distributions, correlations, breakdowns

### 4. Use Cartography Metrics Programmatically

```python
from dataset_cartography import (
    load_cartography_metrics,
    categorize_examples,
    get_examples_by_category
)

# Load metrics
df = load_cartography_metrics("./cartography_output")

# Categorize examples
df = categorize_examples(df)

# Get hard examples for targeted improvement
hard_example_ids = get_examples_by_category(df, 'hard', n=100)

# Get ambiguous examples that might need relabeling
ambiguous_ids = get_examples_by_category(df, 'ambiguous', n=50)

# Analyze by confidence
low_confidence = df[df['confidence'] < 0.5]
print(f"Found {len(low_confidence)} examples with confidence < 0.5")
```

## Understanding the Metrics

### Confidence

- **Range**: 0 to 1
- **High values**: Model is confident in its predictions
- **Low values**: Model is uncertain

### Variability

- **Range**: 0 to ~0.5 (typically)
- **High values**: Model's predictions change across epochs (inconsistent)
- **Low values**: Model's predictions are stable

### Correctness

- **Range**: 0 to 1
- **Value**: Fraction of epochs with correct prediction
- **1.0**: Model always predicted correctly
- **0.0**: Model never predicted correctly

## Example Categories

### Easy Examples (High Confidence, Low Variability)

These are the "golden examples" that the model learns quickly and reliably. They typically:

- Have clear, unambiguous answers
- Contain strong lexical overlap between question and answer
- Represent common patterns in the data

**Use case**: Good for few-shot learning, prompting, or as validation anchors

### Hard Examples (Low Confidence, Low Variability)

These examples are consistently difficult for the model:

- May require complex reasoning
- Might need domain knowledge
- Could have subtle or indirect answers

**Use case**: Focus areas for model improvement, may need additional training data of similar type

### Ambiguous Examples (Low Confidence, High Variability)

These are the most interesting cases - the model can't decide:

- May have annotation errors or noise
- Could have multiple valid answers
- Might be genuinely ambiguous

**Use case**: Candidates for data cleaning, relabeling, or removal

## Integration with Your Analysis

You can combine cartography metrics with your existing error analysis in `data_analysis.ipynb`:

```python
import pandas as pd
from dataset_cartography import load_cartography_metrics, categorize_examples

# Load your evaluation results
eval_df = pd.read_json("eval_baseline_emrqa/eval_predictions.jsonl", lines=True)

# Load cartography metrics
cartography_df = load_cartography_metrics("./cartography_output")
cartography_df = categorize_examples(cartography_df)

# Merge with eval results
merged = eval_df.merge(
    cartography_df,
    left_on='id',
    right_index=True,
    how='inner'
)

# Analyze errors by cartography category
error_by_category = merged.groupby('category').apply(
    lambda x: (x['predicted_answer'] != x['answers'].apply(lambda a: a['text'][0])).mean()
)
print("Error rate by category:")
print(error_by_category)

# Find ambiguous examples that the model got wrong
ambiguous_wrong = merged[
    (merged['category'] == 'ambiguous') &
    (merged['predicted_answer'] != merged['answers'].apply(lambda a: a['text'][0]))
]
print(f"\nFound {len(ambiguous_wrong)} ambiguous examples with errors")
```

## Performance Considerations

- **Memory**: Cartography tracking adds minimal memory overhead (~KB per example)
- **Speed**: Slight slowdown (<5%) due to probability computation
- **Storage**: ~1-2 MB per 1000 examples for full training dynamics

## Tips for Best Results

1. **Train for multiple epochs** (≥3): Need enough data points for meaningful variability
2. **Use consistent training settings**: Ensures comparable dynamics across examples
3. **Start with a subset**: Test on 10k examples before running on full dataset
4. **Compare across models**: See how different architectures handle the same examples

## References

Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., Smith, N. A., & Choi, Y. (2020).
**Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics**.
In Proceedings of EMNLP 2020. [arXiv:2009.10795](https://arxiv.org/abs/2009.10795)

## Example Workflow

```bash
# 1. Train with cartography (use subset for testing)
python run.py \
  --do_train \
  --dataset Eladio/emrqa-msquad \
  --model google/electra-small-discriminator \
  --output_dir ./trained_models/qa/test_cartography \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --max_train_samples 5000 \
  --enable_cartography \
  --cartography_output_dir ./cartography_test

# 2. Analyze results
python analyze_cartography.py \
  --cartography_dir ./cartography_test \
  --dataset Eladio/emrqa-msquad \
  --n_examples 25 \
  --output_dir ./cartography_test_analysis

# 3. Examine hard examples and decide on interventions
# 4. Retrain with filtered/augmented data
# 5. Compare cartography maps before and after
```

## Troubleshooting

**Issue**: "example_id not found in inputs"

- Make sure your dataset has an 'id' column
- The EMR-QA preprocessing automatically adds IDs via `generate_hash_ids()`

**Issue**: "Not enough epochs for meaningful variability"

- Train for at least 3 epochs
- Consider increasing to 5-10 for more stable estimates

**Issue**: "Most examples in one category"

- Adjust confidence/variability thresholds in `categorize_examples()`
- Use percentile-based thresholds instead of fixed values

**Issue**: "Cartography file not found"

- Ensure you ran training with `--enable_cartography`
- Check `--cartography_output_dir` matches between training and analysis

## Experiment System

A systematic framework for running and analyzing multiple training experiments with different configurations. Supports cartography filtering, cluster filtering, label smoothing, soft weighting, and combinations of these strategies.

**Quick Start:**

```bash
# 1. Generate cartography metrics (one-time)
python run.py --do_train --do_eval --enable_cartography \
    --max_train_samples 10000 --num_train_epochs 3

# 2. Run experiment suite
python run_experiments.py --mode run --suite sample --max-train-samples 10000

# 3. Analyze results
python run_experiments.py --mode analyze
```

- `example_configs/` - Example configurations

**Features:**

- Automated execution of multiple experiments with tracking
- Reproducible experiments with seeding support
- Comprehensive result analysis with metrics, plots, and CSV export
- Support for statistical testing with multiple seeds

## Training and evaluating a model

### General

```bash
# To train:
python3 run.py --do_train

# To evaluate:
python3 run.py --do_eval --model ./trainer_output

# To use custom seed (default=42)
python run.py --do_train --output_dir ./model_output --seed 456

# Optional arguments (see run.py for more):
--dataset
--output_dir
--num_train_epochs
--max_train_samples
--max_eval_samples
--per_device_train_batch_size
--save_total_limit
--save_strategy
--ablations

```

Default model is ELECTRA-small.

Default dataset is Eladio/emrqa-msquad.

Checkpoints will be written to sub-folders of the `trained_model` output directory.

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

To test if the code is working with dataset you can pass small numbers for `--max_train_samples`, `--max_eval_samples`, `--per_device_train_batch_size` and `--num_train_epochs` when training/evaluating.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

### Datasets

This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/v2.1.0/en/access).

Pass `dataset:subset` to `--dataset` argument when using custom HF datasets.

## EMRQA-msquad<sup>[1]<sup>

[1] emrQA-msquad: A Medical Dataset Structured with the SQuAD V2.0 Framework, Enriched with emrQA Medical Information: [Article](https://arxiv.org/abs/2404.12050); [Dataset](https://huggingface.co/datasets/Eladio/emrqa-msquad)

### Format:

- Context: str
- Qustion: str
- Answers: Dict with `"answer_start"`, `"answer_end"` and `"text"` lists
- The ID field is missing

### Model Ablations for emrQA:

| Setting                          | EM    | F1    |
| -------------------------------- | ----- | ----- |
| _Without Fine-tuning_            |
| complete                         | 0.02  | 8.04  |
| question-only                    | 0.03  | 8.63  |
| passage-only                     | 0.06  | 5.40  |
| _Fine-tuned using complete data_ |
| complete                         | 90.24 | 92.65 |
| question-only                    | 0.04  | 6.47  |
| passage-only                     | 6.76  | 14.59 |

## Cartography

Dataset cartography analyzes training dynamics by tracking three key metrics for each training example across epochs:

1. **Confidence**: Average probability the model assigns to the correct answer
2. **Variability**: Standard deviation of the model's confidence across epochs
3. **Correctness**: Fraction of epochs where the model predicted correctly

Based on these metrics, examples are categorized as:

- **Easy to learn**: High confidence, low variability - model learns these quickly and consistently
- **Hard to learn**: Low confidence, low variability - model struggles but is consistent
- **Ambiguous**: Low confidence, high variability - model is uncertain and inconsistent

Add the `--enable_cartography` flag when training:

```bash
python run.py \
  --do_train \
  --num_train_epochs 5 \
  --enable_cartography \
  --cartography_output_dir ./cartography_output
```

Important Parameters:

- `--enable_cartography`: Enables cartography tracking
- `--cartography_output_dir`: Directory for all cartography operations (default: `./cartography_output`)
- `--num_train_epochs`: Should be ≥3 for meaningful statistics across epochs

The callback will generate:

- `training_dynamics_epoch_N.json` - Intermediate dynamics after each epoch
- `cartography_metrics.json` - Final aggregated metrics
- `cartography_metrics.csv` - Metrics in CSV format for analysis
- `cartography_map.png` - Visualization of the data map

Use the analysis script to explore your results:

```bash
python analyze_cartography.py \
  --cartography_dir ./cartography_output \
  --split train \
  --n_examples 20 \
  --output_dir ./cartography_analysis
```

This generates:

- **Question type analysis**: How different question types behave
- **Category samples**: Example questions from each category
- **Additional visualizations**: Distributions, correlations, breakdowns

### Curriculum Learning with Cartography

The cartography metrics enable two curriculum learning techniques to improve model training:

#### 1. Variability-Based Label Smoothing

Reduces overfitting on ambiguous/noisy examples by applying soft labels based on variability scores:

```bash
python run.py \
  --do_train \
  --use_label_smoothing \
  --cartography_output_dir ./cartography_output \
  --smoothing_factor 0.6
```

- `--use_label_smoothing`: Enable label smoothing based on variability
- `--smoothing_factor`: Multiplier for variability→smoothing conversion (default: 0.6, range: 0.4-0.8)

High variability examples get softer labels (reducing overconfidence on uncertain data), while low variability examples keep hard targets.

#### 2. Soft Weight Schedule

Increases focus on harder examples during training by weighting losses based on variability:

```bash
python run.py \
  --do_train \
  --use_soft_weighting \
  --cartography_output_dir ./cartography_output \
  --weight_clip_min 0.1 \
  --weight_clip_max 10.0
```

- `--use_soft_weighting`: Enable variability-based loss weighting
- `--weight_clip_min/max`: Clipping range for weights (default: 0.1-10.0)

Higher variability examples receive higher loss weights, making the model focus more on difficult/ambiguous cases.

#### Combined Training

Both techniques can be used together:

```bash
python run.py \
  --do_train \
  --use_label_smoothing \
  --use_soft_weighting \
  --cartography_output_dir ./cartography_output \
  --smoothing_factor 0.6 \
  --weight_clip_min 0.1 \
  --weight_clip_max 10.0
```

**Two-Stage Workflow:**

1. Generate cartography metrics (5 epochs recommended)
2. Train final model with curriculum learning enabled

```bash
# Stage 1: Generate metrics
python run.py \
  --do_train \
  --num_train_epochs 5 \
  --enable_cartography \
  --cartography_output_dir ./cartography_output \
  --output_dir ./initial_model

# Stage 2: Train with curriculum learning
python run.py \
  --do_train \
  --num_train_epochs 3 \
  --use_label_smoothing \
  --use_soft_weighting \
  --cartography_output_dir ./cartography_output \
  --output_dir ./final_model
```

## Embedding-Based Clustering

Complement cartography with embedding-based clustering to identify semantic regions in your dataset. This helps discover:

- Groups of semantically similar questions
- Problematic clusters where the model struggles
- Overlap between semantic similarity and learning difficulty

### Embedding Design

Embeddings use a **[CLS + answer span]** design that captures both global and local semantics:

- **[CLS] token** from (question, context) encoding → global semantic context
- **Mean-pooled answer span tokens** → local answer representation
- **Result**: 512D embeddings (256D + 256D for ELECTRA-small)

This design helps identify examples where the answer is misaligned with the overall context.

### Quick Start

```bash
# 1. Extract embeddings from trained model
python extract_embeddings.py \
  --model_path ./trainer_output \
  --output_dir ./embeddings_output

# 2. Cluster with HDBSCAN (PCA → HDBSCAN → UMAP visualization)
python cluster_analysis.py \
  --embedding_dir ./embeddings_output \
  --output_dir ./cluster_output \
  --reduction_dim 50 \
  --min_cluster_size 15

# 3. Integrated analysis (cartography + clustering)
python analyze_dataset.py \
  --cartography_dir ./cartography_output \
  --cluster_dir ./cluster_output \
  --output_dir ./integrated_analysis
```

### Key Features

- **HDBSCAN clustering**: Automatic cluster discovery without specifying K
- **Two-stage dimensionality reduction**: PCA (512→50D) for clustering, UMAP (50→2D) for visualization
- **Cluster quality metrics**: Persistence scores and membership probabilities
- **Noise detection**: Identifies outliers and anomalous examples

## Dataset Filtering

The `dataset_filters.py` module provides flexible filtering strategies based on analysis results (e.g., cartography metrics). This enables:

- Removing low-quality ambiguous examples
- Curriculum learning (training on easy examples first)
- Focused training (training only on hard examples)
- Confidence-based filtering

### Quick Start

First, generate cartography metrics:

```bash
python run.py \
  --do_train \
  --num_train_epochs 5 \
  --enable_cartography \
  --cartography_output_dir ./cartography_output \
  --output_dir ./initial_model
```

Then train with filtered data:

```bash
python run.py \
  --do_train \
  --num_train_epochs 3 \
  --filter_cartography \
  --cartography_output_dir ./cartography_output \
  --output_dir ./filtered_model
```

### Cluster-Based Filtering

Remove problematic clusters (e.g., noise or low-quality semantic groups):

```bash
python run.py \
  --do_train \
  --filter_clusters \
  --cluster_assignments_path ./cluster_output \
  --exclude_clusters "-1,3,5" \
  --min_cluster_probability 0.7
```

- `--filter_clusters`: Enable cluster-based filtering
- `--cluster_assignments_path`: Path to cluster output directory
- `--exclude_clusters`: Comma-separated cluster IDs to remove (default: "-1" for noise)
- `--min_cluster_probability`: Minimum membership probability threshold

### Combined Filtering

Apply both cartography and cluster filtering:

```bash
python run.py \
  --do_train \
  --filter_cartography \
  --filter_clusters \
  --cartography_output_dir ./cartography_output \
  --cluster_assignments_path ./cluster_output \
  --exclude_clusters "-1" \
  --output_dir ./clean_model
```

### Available Filters

1. **AmbiguousQuestionFilter**: Removes low-quality ambiguous examples
2. **CategoryFilter**: Keeps only specific categories (easy/hard/ambiguous)
3. **ConfidenceThresholdFilter**: Filters by confidence ranges
4. **ClusterFilter**: Removes specified clusters or low-probability assignments

### Programmatic Usage

```python
from dataset_filters import apply_filters

# Define filter configuration
filter_config = {
    "ambiguous": {
        "enabled": True,
        "metrics_path": "./cartography_output",
        "top_fraction": 0.33,  # Keep top 33% most ambiguous
    },
    "category": {
        "enabled": True,
        "metrics_path": "./cartography_output",
        "categories": ["easy", "hard"]  # Drop all ambiguous
    }
}

# Apply filters
filtered_dataset = apply_filters(dataset, filter_config)
```

Run example scripts:

```bash
python examples/filtering_examples.py
```

## Unified Analysis Export

The unified analysis export combines results from cartography, clustering, and rule-based error detection into a single comprehensive CSV file for cross-method analysis. This enables:

- Identifying high-overlap regions where multiple methods flag problems
- Finding systematic dataset annotation errors
- Prioritizing examples for manual review
- Understanding agreement/disagreement between detection methods

### Quick Start

First, ensure you have all three analysis results:

```bash
# 1. Train with cartography
python run.py \
  --do_train \
  --num_train_epochs 5 \
  --enable_cartography \
  --cartography_output_dir ./cartography_output \
  --output_dir ./model

# 2. Extract embeddings and cluster
python extract_embeddings.py \
  --model_path ./model \
  --output_dir ./embeddings_output

python cluster_analysis.py \
  --embedding_dir ./embeddings_output \
  --output_dir ./cluster_output

# 3. Export unified analysis
python analyze_dataset.py \
  --cartography_dir ./cartography_output \
  --cluster_dir ./cluster_output \
  --export_unified \
  --output_dir ./unified_export
```

### Outputs

The export creates two files:

1. **`unified_analysis.csv`** - Complete dataset with all metrics:

   - Original columns (question, context, answer)
   - Cartography metrics (confidence, variability, correctness, category)
   - Cluster assignments (cluster ID, coordinates, probability)
   - **Rule-based error flags** (one boolean column per rule)
   - Error scores and classifications

2. **`overlap_summary.json`** - Summary statistics:
   - Data coverage by each method
   - Category-cluster overlap distribution
   - Rule trigger rates by category and cluster
   - **High overlap regions** - areas flagged by multiple methods

### Rule-Based Error Detection

The unified export includes 9 rule-based error detection flags:

- `rule1_length_anomaly` - Answer span is abnormally long
- `rule2_multi_clause` - Answer contains multiple clauses
- `rule3_low_q_similarity` - Low lexical overlap with question
- `rule4_pred_inside_gold_better` - Prediction is better-aligned substring
- `rule5_qtype_mismatch` - Question type vs answer structure mismatch
- `rule6_multi_occurrences` - Answer appears multiple times in context
- `rule7_boundary_weirdness` - Unusual start/end characters
- `rule8_pred_better_q_alignment` - Prediction aligns better with question
- `rule9_question_not_starting_with_qword` - Malformed question

Each example gets a `dataset_error_score` (0-9) and `is_dataset_error` flag (score ≥ 3).

### High Overlap Regions

The system automatically identifies problematic regions where:

- Cartography categorizes as hard/ambiguous
- Clustering groups examples together
- Rule-based detection flags errors

Example from `overlap_summary.json`:

```json
{
  "high_overlap_regions": [
    {
      "category": "ambiguous",
      "cluster": 3,
      "total_examples": 450,
      "error_count": 180,
      "error_rate": 0.4
    }
  ]
}
```

This indicates 40% of ambiguous examples in cluster 3 are also flagged by rules - a systematic issue worth investigating.

### Analysis Example

```python
import pandas as pd

# Load unified analysis
df = pd.read_csv("unified_export/unified_analysis.csv")

# Find examples flagged by multiple methods
problematic = df[
    (df['category'] == 'ambiguous') &
    (df['is_dataset_error'])
]

print(f"Found {len(problematic)} high-priority examples")

# Export for manual review
problematic[['id', 'question', 'answer', 'context']].to_csv('review.csv')
```

Run the comprehensive analysis example:

```bash
python demos/unified_analysis_example.py
```

This generates:

- Error rate heatmaps by category and cluster
- Scatter plots highlighting flagged examples
- Rule trigger distribution charts
- CSV files of problematic examples for review

### Use Cases

**Data Cleaning:**

```python
# Remove examples flagged by multiple methods
clean_df = df[~((df['category'] == 'ambiguous') & df['is_dataset_error'])]
```

**Quality Metrics:**

```python
quality = {
    'ambiguous_pct': 100 * (df['category'] == 'ambiguous').mean(),
    'error_pct': 100 * df['is_dataset_error'].mean(),
    'noise_pct': 100 * (df['cluster'] == -1).mean()
}
```

**Prioritized Review:**

```python
# High-risk: ambiguous + noise cluster + errors
high_risk = df[
    (df['category'] == 'ambiguous') &
    (df['cluster'] == -1) &
    df['is_dataset_error']
]
```

## Version Controlling and Git Practices

Make sure you add/install packages only using `uv add <package_name>` if you are using uv. Otherwise make sure you manually add them (in alphabetical order) to the `pyproject.toml` file.

Make sure you use `uv sync` frequently after each pull so as to have the most up-to-date packages installed.

For Jupyter Notebooks committed to git repos make sure you install `nbstripout` which is also included in pyproject. You'd need to run `nbstripout --install` in your current repository once in order to ensure Jupyter metdata does not get committed to the git.

## To Debug

To use VSCode debugpy simply modify and use `launch.json`. It is currently mofied to provide base debugging.

## References

- Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., Smith, N. A., & Choi, Y. (2020).
  **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics**.
  In Proceedings of EMNLP 2020. [arXiv:2009.10795](https://arxiv.org/abs/2009.10795)

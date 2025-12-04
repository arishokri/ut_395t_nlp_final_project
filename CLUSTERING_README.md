# HDBSCAN Clustering for Dataset Analysis

HDBSCAN-based clustering system for identifying semantic patterns in QA datasets, integrated with dataset cartography.

## Quick Start

```bash
# 1. Extract embeddings from trained model
python extract_embeddings.py \
  --model_path ./trainer_output \
  --output_dir ./embeddings

# 2. Cluster with HDBSCAN (PCA → HDBSCAN → UMAP)
python cluster_analysis.py \
  --embedding_dir ./embeddings \
  --output_dir ./clusters \
  --reduction_dim 50 \
  --min_cluster_size 15

# 3. Integrate with cartography
python analyze_dataset.py \
  --cartography_dir ./cartography_output \
  --cluster_dir ./clusters \
  --output_dir ./analysis
```

## Pipeline Overview

```
Embeddings ([CLS + answer span], 512D for ELECTRA-small)
    ↓
PCA Reduction (→ 30-50D)
    ↓
HDBSCAN Clustering
    ↓
UMAP Reduction (→ 2D)
    ↓
Visualization + Analysis
```

## Embedding Design

**Representation = [CLS of (Q, context)] + mean-pooled answer span tokens**

This captures:

- **Global semantics**: [CLS] token from question-context encoding
- **Local answer**: Mean-pooled answer span tokens
- **Alignment**: How well the answer fits the overall semantic context

## Key Features

- **HDBSCAN Clustering**: Automatic cluster discovery without specifying K
- **Hierarchical**: Builds cluster hierarchy with stability metrics
- **Soft Assignments**: Provides membership probabilities
- **Noise Detection**: Identifies outliers/anomalous examples
- **Integrated Analysis**: Combines with dataset cartography

## Key Parameters

| Parameter            | Default   | Recommended Range | Effect                      |
| -------------------- | --------- | ----------------- | --------------------------- |
| `--reduction_dim`    | 50        | 30-50             | Intermediate dimensionality |
| `--min_cluster_size` | 10        | 10-25             | Minimum cluster size        |
| `--min_samples`      | None      | 10-30             | Clustering conservativeness |
| `--metric`           | euclidean | euclidean, cosine | Distance metric             |

## Output Files

```
cluster_output/
├── cluster_assignments.csv      # Cluster ID + probability per example
├── cluster_statistics.csv        # Size, persistence per cluster
├── cluster_samples.json          # Example questions per cluster
├── cluster_visualization.png     # 2D scatter plot
├── cluster_persistence.png       # HDBSCAN stability scores
└── cluster_probabilities.png     # Membership distributions
```

## Embedding Details

For ELECTRA-small (hidden_size=256):

- [CLS] component: 256 dimensions
- Answer span component: 256 dimensions
- Total embedding: 512 dimensions

The two components can be analyzed separately to understand answer-context alignment.

## Example Workflow

See `full_workflow.sh` for a complete end-to-end example including:

1. Training with cartography
2. Embedding extraction
3. Multiple clustering configurations
4. Integrated analysis

```bash
bash full_workflow.sh
```

## Parameter Tuning Guide

### Too Many Small Clusters?

```bash
--reduction_dim 30 \
--min_cluster_size 20 \
--min_samples 25
```

### Too Much Noise?

```bash
--reduction_dim 50 \
--min_cluster_size 10 \
--min_samples 10
```

### Want Semantic Clustering?

```bash
--reduction_dim 40 \
--metric cosine \
--min_cluster_size 15
```

### More Conservative Clustering?

```bash
--reduction_dim 30 \
--min_cluster_size 25 \
--min_samples 30
```

## Interpreting Cluster Probabilities

| Probability | Meaning       | Action                         |
| ----------- | ------------- | ------------------------------ |
| > 0.8       | Core member   | Use for cluster interpretation |
| 0.5-0.8     | Peripheral    | Examine for insights           |
| < 0.5       | Boundary case | Potential data quality issue   |
| -1 (noise)  | Outlier       | May need special handling      |

## Interpreting Cluster Persistence

| Persistence | Meaning        | Action                            |
| ----------- | -------------- | --------------------------------- |
| > 0.5       | Stable cluster | Trust this cluster                |
| 0.3-0.5     | Moderate       | Examine samples carefully         |
| < 0.3       | Weak           | May be artifact, verify semantics |

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
- `--cartography_output_dir`: Where to save cartography outputs (default: `./cartography_output`)
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

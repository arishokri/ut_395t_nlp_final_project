# fp-dataset-artifacts

Project by Kaj Bostrom, Jifan Chen, and Greg Durrett. Code by Kaj Bostrom and Jifan Chen.

## Training and evaluating a model

```bash
# To train:

python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/

# To evaluate:

`python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/`

# With specific parameters
python run.py --do_train --task qa --output_dir ./trained_models/qa/squad --num_train_epochs 3 --per_device_train_batch_size 16

```

Default model is ELECTRA-small.

Default dataset for nli is SNLI and for qa is SQUaD.

Checkpoints will be written to sub-folders of the `trained_model` output directory.

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

To test if the code is working with dataset you can pass small numbers for `--max_train_samples`, `--max_eval_samples`, `--per_device_train_batch_size` and `--num_train_epochs` when training/evaluating.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

## Datasets

This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/v2.1.0/en/access).

Pass `dataset:subset` to `--dataset` argument when using custom HF datasets.

### SQuAD (Default)

**Format:**

- Context: Plain string
- Answers: Dict with `"text"` and `"answer_start"` lists
- ID: Provided in dataset

### HotpotQA

**Format:**

- Context: Nested dict with `"sentences"` (list of lists) and `"title"` fields
- Answer: Single string (no `answer_start`)
- ID: Provided in dataset

**Processing:**

- Contexts are automatically flattened from nested structure
- Answer spans are automatically located in the context
- Falls back to case-insensitive search if exact match fails

## Results

**NLI Task**

| setup | dataset           | model         | train time\* | accuracy |
| ----- | ----------------- | ------------- | ------------ | -------- |
| Base  | SNLI              | Electra-small | 50m          | 89.23%   |
| Base  | mNLI<sup>[1]<sup> | Electra-small | 36m          | 81.61%   |

**QA Task**

| setup | dataset                   | model         | train time\* | train time\*\* | EM    | F1    |
| ----- | ------------------------- | ------------- | ------------ | -------------- | ----- | ----- |
| Base  | SQuAD                     | Electra-small | 24m          | 17m            | 78.20 | 86.24 |
| Base  | HotpotQA                  | Electra-small | NA           | NA             | NA    | NA    |
| Base  | emrQA-msquad<sup>[2]<sup> | Electra-small | 1h:10m       | 50m            | 90.24 | 92.65 |

\* train time on RTX 3090

\*\* train time on RTX 5070 Ti

[1] GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding: [Article](https://arxiv.org/abs/1804.07461); [Dataset](https://huggingface.co/datasets/nyu-mll/glue)

    Use with glue:mnli

[2] emrQA-msquad: A Medical Dataset Structured with the SQuAD V2.0 Framework, Enriched with emrQA Medical Information: [Article](https://arxiv.org/abs/2404.12050); [Dataset](https://huggingface.co/datasets/Eladio/emrqa-msquad)

## To Debug

To use VSCode debugpy simply modify and use `launch.json`. It is currently mofied to provide base debugging.

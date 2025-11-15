# fp-dataset-artifacts

Project by Kaj Bostrom, Jifan Chen, and Greg Durrett. Code by Kaj Bostrom and Jifan Chen.

## Getting Started

You'll need Python >= 3.6 to run the code in this repo.

First, clone the repository:

`git clone git@github.com:gregdurrett/fp-dataset-artifacts.git`

Then install the dependencies:

`pip install --upgrade pip`

`pip install -r requirements.txt`

If you're running on a shared machine and don't have the privileges to install Python packages globally,
or if you just don't want to install these packages permanently, take a look at the "Virtual environments"
section further down in the README.

To make sure pip is installing packages for the right Python version, run `pip --version`
and check that the path it reports is for the right Python interpreter.

## Training and evaluating a model

To train an ELECTRA-small model on the SNLI natural language inference dataset, you can run the following command:

`python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/`

Checkpoints will be written to sub-folders of the `trained_model` output directory.
To evaluate the final trained model on the SNLI dev set, you can use

`python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/`

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

## Working with datasets

This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/v2.1.0/en/access).

Pass `dataset:subset` to `--dataset` argument when using custom HF datasets.

## Results

**NLI Task**

| setup | dataset           | model         | train time\* | accuracy |
| ----- | ----------------- | ------------- | ------------ | -------- |
| Base  | SNLI              | Electra-small | 50m          | 89.23%   |
| Base  | mNLI<sup>[1]<sup> | Electra-small | 36m          | 81.61%   |

**QA Task**

| setup | dataset                   | model         | train time\* | train time\** | EM     | F1     |     
| ----- | ------------------------- | ------------- | ------------ | ------------- | ------ | ------ |
| Base  | SQuAD                     | Electra-small | 24m          | 17m           | 78.20  | 86.24  |
| Base  | HotpotQA                  | Electra-small | NA           | NA            | NA     | NA     |
| Base  | emrQA-msquad<sup>[2]<sup> | Electra-small | 1h:10m       | 50m           | 90.24  | 92.65  |

\* train time on RTX 3090
\** train time on RTX 5070 Ti

[1] GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding: [Article](https://arxiv.org/abs/1804.07461); [Dataset](https://huggingface.co/datasets/nyu-mll/glue)

    Use with glue:mnli

[2] emrQA: A Large Corpus for Question Answering on Electronic Medical Records: [Article](https://arxiv.org/abs/1809.00732); [Dataset](https://huggingface.co/datasets/Eladio/emrqa-msquad) needs verification

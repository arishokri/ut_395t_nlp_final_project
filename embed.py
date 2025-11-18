import itertools
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")

# ==========================================
# 1. Dataclasses for HF-style CLI arguments
# ==========================================


@dataclass
class ModelArguments:
    """
    Arguments about which model to use.
    """

    model: str = field(
        metadata={
            "help": "Model identifier (e.g. microsoft/mpnet-base, LLukas22/all-mpnet-base-v2-embedding-all, or a local path."
        }
    )
    sentence_transformer: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, treat `model` as a SentenceTransformer model. "
                "If False, treat it as a plain HF encoder/MLM model."
            )
        },
    )


@dataclass
class DataArguments:
    """
    Arguments about the data and task.
    """

    # Financial news (MLM)
    train_dataset: str = field(
        default="Brianferrell787/financial-news-multisource",
        metadata={"help": "Name of the financial news dataset on the Hub."},
    )
    train_dataset_files: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional comma-separated data_files patterns inside the dataset. "
                "E.g. 'data/fnspid_news/*.parquet,data/yahoo_finance_articles/*.parquet'. "
                "If None, use the dataset's default."  # dataset default split
            )
        },
    )
    max_train_rows: int = field(
        default=50000,
        metadata={"help": "Max number of rows to sample for MLM pretraining."},
    )
    validation_split_ratio: float = field(
        default=0.05,
        metadata={"help": "Ratio of news data to use as validation for MLM."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length for MLM tokenization."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Masking probability for MLM."},
    )

    # FiQA evaluation
    eval_dataset: str = field(
        default="LLukas22/fiqa",
        metadata={
            "help": "Name of the sentence similarity dataset used for evaluation."
        },
    )
    max_eval_rows: int = field(
        default=5000,
        metadata={"help": "Max number of examples to use for retrieval eval."},
    )


# ===========================
# 2. Retrieval metrics helper
# ===========================


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


def compute_retrieval_metrics(
    q_emb: torch.Tensor,  # [N, d]
    a_emb: torch.Tensor,  # [N, d]
) -> RetrievalMetrics:
    """
    Simple retrieval setup: each question_i should retrieve answer_i
    from the pool of all answers.
    """
    q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1)
    a_emb = torch.nn.functional.normalize(a_emb, p=2, dim=1)

    sim = q_emb @ a_emb.T  # [N, N]
    N = sim.size(0)

    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    for i in range(N):
        row = sim[i]
        sorted_idx = torch.argsort(row, descending=True)
        pos = (sorted_idx == i).nonzero(as_tuple=False).item()
        rank = pos + 1  # 1-based
        ranks.append(rank)
        if rank == 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1

    mrr = sum(1.0 / r for r in ranks) / N
    return RetrievalMetrics(
        recall_at_1=hits_at_1 / N,
        recall_at_5=hits_at_5 / N,
        recall_at_10=hits_at_10 / N,
        mrr=mrr,
    )


# =================================
# 3. Dataset loading / preparation
# =================================


def load_fiqa(data_args: DataArguments):
    """
    Load FiQA question/answer pairs.
    """
    ds = load_dataset(data_args.eval_dataset)
    ds = ds["train"]

    if data_args.max_eval_rows is not None:
        n = min(data_args.max_eval_rows, len(ds))
        ds = ds.select(range(n))

    questions = ds["question"]
    answers = ds["answer"]
    return questions, answers


def sample_financial_news(data_args: DataArguments):
    """
    Stream from the financial news dataset and gather a small in-memory sample
    for MLM pretraining.
    """
    if data_args.train_dataset_files is not None:
        files = [
            p.strip() for p in data_args.train_dataset_files.split(",") if p.strip()
        ]
    else:
        files = None

    streaming_ds = load_dataset(
        data_args.train_dataset,
        data_files=None if files is None else {"train": files},
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    print(
        f"Streaming financial news from {data_args.train_dataset}, "
        f"collecting up to {data_args.max_train_rows} rows..."
    )
    samples = list(itertools.islice(streaming_ds, data_args.max_train_rows))
    print(f"Collected {len(samples)} rows")

    raw = Dataset.from_list(samples)
    split = raw.train_test_split(test_size=data_args.validation_split_ratio, seed=42)
    return split["train"], split["test"]


# ===========================
# 4. Embedding helpers
# ===========================


def encode_with_sentence_transformer(
    texts: List[str],
    model_name: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Use a SentenceTransformer model for embeddings.
    """
    st_model = SentenceTransformer(model_name_or_path=model_name, device=DEVICE)
    emb = st_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return torch.from_numpy(emb)


def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling over token embeddings (Sentence-BERT style).
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_with_hf_encoder(
    texts: List[str],
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """DEVICE
    Use a plain HF encoder model with mean pooling to get sentence embeddings.
    """
    encoder.eval()
    encoder.to(DEVICE)

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            outputs = encoder(**toks)
            pooled = mean_pooling(outputs.last_hidden_state, toks["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.cpu())

    return torch.cat(all_embs, dim=0)


# ==============================
# 5. MLM training (do_train)
# ==============================


def tokenize_mlm(
    examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_length: int
):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def run_mlm_training(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
):
    """
    Continue MLM pretraining on financial news.
    """
    # 1) Data
    train_raw, val_raw = sample_financial_news(data_args)

    # 2) Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model, use_fast=True)
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_args.model)

    # 3) Tokenize
    def _tok_fn(batch):
        return tokenize_mlm(batch, tokenizer, max_length=data_args.max_seq_length)

    train_tok = train_raw.map(
        _tok_fn,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train financial news",
    )
    val_tok = val_raw.map(
        _tok_fn,
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Tokenizing val financial news",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability,
    )

    # 4) Trainer
    trainer = Trainer(
        model=mlm_model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok if training_args.eval_strategy != "no" else None,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("Training finished.")
    print(f"Model saved to {training_args.output_dir}")
    return train_result


# ==============================
# 6. FiQA evaluation (do_eval)
# ==============================


def run_fiqa_eval(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
):
    """
    Evaluate FiQA retrieval with the specified model.
    - If sentence_transformer=True: treat model as SentenceTransformer.
    - Else: treat model as HF encoder (e.g., MLM-continued mpnet).
    """
    questions, answers = load_fiqa(data_args)

    batch_size = (
        training_args.per_device_eval_batch_size
        if training_args.per_device_eval_batch_size is not None
        else 64
    )

    if model_args.sentence_transformer:
        print(f"Using SentenceTransformer model: {model_args.model}")
        q_emb = encode_with_sentence_transformer(
            questions,
            model_args.model,
            batch_size=batch_size,
        )
        a_emb = encode_with_sentence_transformer(
            answers,
            model_args.model,
            batch_size=batch_size,
        )
    else:
        print(f"Using HF encoder model: {model_args.model}")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model, use_fast=True)
        encoder = AutoModel.from_pretrained(model_args.model)

        q_emb = encode_with_hf_encoder(
            questions,
            encoder,
            tokenizer,
            batch_size=batch_size,
            max_length=data_args.max_seq_length,
        )
        a_emb = encode_with_hf_encoder(
            answers,
            encoder,
            tokenizer,
            batch_size=batch_size,
            max_length=data_args.max_seq_length,
        )

    print("Computing FiQA retrieval metrics...")
    metrics = compute_retrieval_metrics(q_emb, a_emb)
    print("FiQA retrieval results:")
    print(f"  Recall@1  = {metrics.recall_at_1:.4f}")
    print(f"  Recall@5  = {metrics.recall_at_5:.4f}")
    print(f"  Recall@10 = {metrics.recall_at_10:.4f}")
    print(f"  MRR       = {metrics.mrr:.4f}")

    return metrics


# ==============================
# 7. Main entry: HfArgumentParser
# ==============================


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train:
        print("=== Running MLM pretraining on financial news (do_train=True) ===")
        run_mlm_training(model_args, data_args, training_args)

    if training_args.do_eval:
        print("=== Running FiQA retrieval evaluation (do_eval=True) ===")
        run_fiqa_eval(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

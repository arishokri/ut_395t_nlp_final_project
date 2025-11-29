import collections
from collections import OrderedDict, defaultdict
from typing import Tuple

import random
import numpy as np
from tqdm.auto import tqdm
from transformers import EvalPrediction, Trainer
from transformers.trainer_utils import PredictionOutput

QA_MAX_ANSWER_LENGTH = 30

FILLER_WORDS = [
    "the", "patient", "may", "have", "no", "significant", "history",
    "of", "current", "presents", "with", "possible", "likely", "reports",
    "denies", "for", "and", "or", "is"
]

def normalize_context_hotpotqa(context):
    """Convert HotpotQA nested context format to a single string."""
    if isinstance(context, dict) and "sentences" in context:
        combined_sent = []
        for sent_list in context["sentences"]:
            combined = "\n".join(s.lstrip() for s in sent_list)
            combined_sent.append(combined)
        return "\n".join(c for c in combined_sent)
    return context


def normalize_answer_hotpotqa(answer_text, context):
    """Convert HotpotQA answer format (single string) to SQuAD format (dict with lists)."""
    # Find the answer span in the context
    answer_start = context.find(answer_text)
    if answer_start == -1:
        # If exact match not found, try case-insensitive search
        answer_start = context.lower().find(answer_text.lower())

    return {
        "text": [answer_text] if answer_start != -1 else [],
        "answer_start": [answer_start] if answer_start != -1 else [],
    }

def preprocess_dataset_for_qa(examples, dataset_name=None):
    """
    Preprocess examples based on dataset format.
    Returns: (contexts, normalized_answers)

    Supported datasets:
    - "squad" (default): standard SQuAD format
    - "hotpotqa/hotpot_qa": HotpotQA format with nested contexts and single answer string
    - "Eladio/emrqa-msquad": EMR-QA format
    """
    # Handle context normalization
    if dataset_name and "hotpotqa" in dataset_name.lower():
        # HotpotQA: nested context structure
        contexts = [normalize_context_hotpotqa(ctx) for ctx in examples["context"]]
    else:
        # SQuAD and EMR-QA: context is already a string
        contexts = examples["context"]

    # Handle answer normalization
    if dataset_name and "hotpotqa" in dataset_name.lower():
        # HotpotQA: single answer string, need to find answer_start
        normalized_answers = [
            normalize_answer_hotpotqa(answer_text, contexts[i])
            for i, answer_text in enumerate(examples["answer"])
        ]
    elif "answers" in examples:
        # SQuAD and EMR-QA: already have answer_start
        normalized_answers = examples["answers"]
    elif "answer" in examples:
        # Fallback: single answer field without dataset-specific handling
        normalized_answers = [
            normalize_answer_hotpotqa(answer_text, contexts[i])
            for i, answer_text in enumerate(examples["answer"])
        ]
    else:
        raise ValueError("Dataset must have either 'answer' or 'answers' field")

    return contexts, normalized_answers


def prepare_dataset_for_evaluation(examples, dataset_name=None):
    """
    Prepare context for evaluation based on dataset format.
    Returns: contexts (list of strings)
    """
    if dataset_name and "hotpotqa" in dataset_name.lower():
        # HotpotQA: nested context structure
        contexts = [normalize_context_hotpotqa(ctx) for ctx in examples["context"]]
    else:
        # SQuAD and EMR-QA: context is already a string
        contexts = examples["context"]

    return contexts


def normalize_answers_for_metrics(example, dataset_name=None):
    """
    Normalize answers for metric computation based on dataset format.
    Returns: dict with "text" and "answer_start" lists
    """
    if dataset_name and "hotpotqa" in dataset_name.lower():
        # HotpotQA format: single answer string
        return {
            "text": [example["answer"]],
            "answer_start": [0],  # Dummy value for compatibility
        }
    elif "answers" in example:
        ans = example["answers"]
        # Case 1: dict with lists: {"text": [...], "answer_start": [...]}
        if isinstance(ans, dict):
            return {
                "text": ans.get("text", []),
                "answer_start": ans.get("answer_start", []),
            }
        # Case 2: list of dicts: [{"text": ..., "answer_start": ..., ...}, ...]
        else:
            texts = []
            starts = []
            for a in ans:
                texts.append(a.get("text", ""))
                starts.append(a.get("answer_start", 0))
            return {"text": texts, "answer_start": starts}
    else:
        raise ValueError("Example must have either 'answer' or 'answers' field")


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = (
        tokenizer.model_max_length if max_seq_length is None else max_seq_length
    )

    tokenized_examples = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    tokenized_examples["label"] = examples["label"]
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        "accuracy": (np.argmax(eval_preds.predictions, axis=1) == eval_preds.label_ids)
        .astype(np.float32)
        .mean()
        .item()
    }

# Attempt 2: 
# >>> Question-only ablation: randomize contexts & add filler if needed <<<
# def randomize_contexts_by_cyclic_shift(contexts, filler_words, seed: int = 42):
#     n = len(contexts)
#     rng = random.Random(seed)

#     indices = list(range(n))
#     rng.shuffle(indices)  # random permutation: i -> indices[i]

#     new_contexts = []
#     for i in range(n):
#         original = contexts[i]
#         target_len = len(original)
#         src = contexts[indices[i]]

#         ctx = src
#         if len(ctx) < target_len:
#             while len(ctx) < target_len:
#                 filler = rng.choice(filler_words)
#                 if ctx:
#                     ctx += " "
#                 ctx += filler
#             ctx = ctx[:target_len]
#         elif len(ctx) > target_len:
#             ctx = ctx[:target_len]

#         new_contexts.append(ctx)

#     return new_contexts

# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(
    examples,
    tokenizer,
    qa_mode: str,
    dataset_name=None,
    max_seq_length=None,
):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length

    # If passage-only, destroy question content
    if qa_mode == "p_only":
        # generic question template so model doesn't find value in this
        questions = ["What is the answer?" for _ in questions]

    # Preprocess based on dataset format
    contexts, normalized_answers = preprocess_dataset_for_qa(examples, dataset_name)

    # comment out the question only type implementation to use
    # # >>> NEW: randomized-context variant for q_only! <<<
    # if qa_mode == "q_only":
    #     # replace each context with another record's context,
    #     # keeping the same length via pad/truncate
    #     contexts = randomize_contexts_by_cyclic_shift(contexts, FILLER_WORDS, seed=42)
    #     # NOTE: we are *not* changing normalized_answers here

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # >>> Question-only ablation: mask out context tokens <<<
    if qa_mode == "q_only" and "token_type_ids" in tokenized_examples:
        pad_id = tokenizer.pad_token_id
        input_ids = tokenized_examples["input_ids"]
        attention_mask = tokenized_examples["attention_mask"]
        token_type_ids = tokenized_examples["token_type_ids"]

        for i in range(len(input_ids)):
            ids_row = input_ids[i]
            attn_row = attention_mask[i]
            seg_row = token_type_ids[i]

            for j in range(len(ids_row)):
                # In BERT/ELECTRA-style tokenizers: 0 = question, 1 = context
                if seg_row[j] == 1 and attn_row[j] == 1:
                    attn_row[j] = 0  # hide from attention
                    ids_row[j] = pad_id  # optional: visually mark as PAD

        tokenized_examples["input_ids"] = input_ids
        tokenized_examples["attention_mask"] = attention_mask
    # <<< end question-only ablation >>>


    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = normalized_answers[sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            if "answer_end" in answers and len(answers["answer_end"]) > 0:
                end_char = answers["answer_end"][0]
            else:
                end_char = start_char + len(answers["text"][0])
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(
    examples,
    tokenizer,
    qa_mode: str,
    dataset_name=None,
):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length

    # Preprocess based on dataset format
    contexts = prepare_dataset_for_evaluation(examples, dataset_name)

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    dataset_name=None,
    n_best_size: int = 20,
):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        )
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features."
        )

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        # Handle different context formats based on dataset
        if dataset_name and "hotpotqa" in dataset_name.lower():
            context = normalize_context_hotpotqa(example["context"])
        else:
            context = example["context"]

        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, dataset_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.dataset_name = dataset_name

    def evaluate(
        self,
        eval_dataset=None,  # denotes the dataset after mapping
        eval_examples=None,  # denotes the raw dataset
        ignore_keys=None,  # keys to be ignored in dataset
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds = postprocess_qa_predictions(
                eval_examples, eval_dataset, output.predictions, self.dataset_name
            )
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in eval_preds.items()
            ]

            # Normalize answers based on dataset format
            references = []
            for ex in eval_examples:
                cleaned_answers = normalize_answers_for_metrics(ex, self.dataset_name)
                references.append({"id": ex["id"], "answers": cleaned_answers})

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions, label_ids=references)
            )

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

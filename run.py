import json
import os

import datasets
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from helpers import (
    QuestionAnsweringTrainer,
    compute_metrics,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
)

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path, default="trainer_output/">
    #     Where to put the trained model checkpoint(s) and any eval predictions.

    argp.add_argument(
        "--model",
        type=str,
        default="google/electra-small-discriminator",
        help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""",
    )
    argp.add_argument(
        "--dataset",
        type=str,
        default="Eladio/emrqa-msquad",
        help="""This argument overrides the default dataset.""",
    )
    argp.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""",
    )
    argp.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of examples to train on.",
    )
    argp.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit the number of examples to evaluate on.",
    )
    # Additional parameter for the question only model training
    argp.add_argument(
        "--ablations",
        type=str,
        choices=["none", "q_only", "p_only"],
        default="none",
        help="If se to q_only, only uses questions, if set to p_only only uses passages. Otherwise uses full sentence.",
    )

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    dataset_name = args.dataset
    if args.dataset.endswith(".json") or args.dataset.endswith(".jsonl"):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset("json", data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = "train"
    else:
        dataset_id = tuple(args.dataset.split(":"))
        eval_split = (
            "validation_matched" if dataset_id == ("glue", "mnli") else "validation"
        )
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)

    # TODO: Is this the correct way of initializing this class?
    model_class = AutoModelForQuestionAnswering  # Fine-tuning head for QA task.
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ablations = args.ablations
    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    # TODO: Get rid of these lambda functions and rename the main function to remove qa.
    prepare_train_dataset = lambda exs: prepare_train_dataset_qa(
        exs, tokenizer, ablations
    )
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    print(
        "Preprocessing data... (this takes a little bit, should only happen once per dataset)"
    )

    # Dataset-specific preprocessing
    # TODO: make sure this id is deterministic. Maybe use hashing. Important for Cartography
    # TODO: exact match for dataset name here.
    if "emrqa" in dataset_name.lower():
        # EMR-QA: Add ID column if missing
        for split in dataset.keys():
            if "id" not in dataset[split].column_names:
                dataset[split] = dataset[split].map(
                    lambda ex, idx: {"id": str(idx)}, with_indices=True
                )

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset["train"]
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
        )

    # Select the training configuration
    trainer_kwargs = {}
    eval_kwargs = {}
    # For an example of a valid compute_metrics function, see compute_metrics in helpers.py.
    # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
    # to enable the question-answering specific evaluation metrics
    eval_kwargs["eval_examples"] = eval_dataset

    # TODO: Why are we doing this?
    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
        **trainer_kwargs,
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see helpers).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print("Evaluation results:")
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(
            os.path.join(training_args.output_dir, "eval_metrics.json"),
            encoding="utf-8",
            mode="w",
        ) as f:
            json.dump(results, f)

        with open(
            os.path.join(training_args.output_dir, "eval_predictions.jsonl"),
            encoding="utf-8",
            mode="w",
        ) as f:
            predictions_by_id = {
                pred["id"]: pred["prediction_text"]
                for pred in eval_predictions.predictions
            }
            for example in eval_dataset:
                example_with_prediction = dict(example)
                example_with_prediction["predicted_answer"] = predictions_by_id[
                    example["id"]
                ]
                f.write(json.dumps(example_with_prediction))
                f.write("\n")


if __name__ == "__main__":
    main()

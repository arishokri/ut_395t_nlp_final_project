import json
import os
from functools import partial

import datasets
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from dataset_cartography import DatasetCartographyCallback
from helpers import (
    DataCollatorWithExampleId,
    QuestionAnsweringTrainer,
    compute_metrics,
    generate_hash_ids,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
)

from rule_based_errors import rule9_question_not_starting_with_qword
from analyze_cartography import load_cartography_metrics, categorize_examples

NUM_PREPROCESSING_WORKERS = 2

def filter_ambiguous_non_questions(dataset, cartography_metrics_path):
    """
    Filter out examples that are both:
    1. Categorized as 'ambiguous' by dataset cartography
    2. Identified as non-questions by rule9_question_not_starting_with_qword
    """
    try:
        # Load cartography metrics
        cartography_df = load_cartography_metrics(cartography_metrics_path)
        cartography_df = categorize_examples(cartography_df)
        
        # Get ambiguous example IDs
        ambiguous_ids = set(cartography_df[cartography_df['category'] == 'ambiguous'].index.tolist())
        
        # Convert dataset to pandas for easier manipulation
        df = dataset.to_pandas()
        
        # Identify examples to remove (ambiguous AND non-questions)
        examples_to_remove = []
        
        for idx, row in df.iterrows():
            example_id = row.get('id')
            question = row.get('question', '')
            
            # Check if example is ambiguous AND a non-question
            if (example_id in ambiguous_ids and 
                rule9_question_not_starting_with_qword(question)):
                examples_to_remove.append(idx)
        
        # Filter out the flagged examples
        if examples_to_remove:
            filtered_df = df.drop(examples_to_remove)
            # Convert back to datasets format
            from datasets import Dataset
            filtered_dataset = Dataset.from_pandas(filtered_df)
            
            print(f"Removed {len(examples_to_remove)} examples that were both ambiguous and non-questions")
            print(f"Original size: {len(df)}, Filtered size: {len(filtered_df)}")
            
            return filtered_dataset
        else:
            print("No examples found that are both ambiguous and non-questions")
            return dataset
            
    except Exception as e:
        print(f"Error loading cartography metrics: {e}")
        print("Skipping filtering...")
        return dataset
    
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
        default=512,
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
    argp.add_argument(
        "--enable_cartography",
        action="store_true",
        help="Enable dataset cartography tracking during training to identify easy/hard/ambiguous examples.",
    )
    argp.add_argument(
        "--cartography_output_dir",
        type=str,
        default="./cartography_output",
        help="Directory to save cartography outputs.",
    )
    argp.add_argument(
        "--filter_ambiguous_non_questions",
        action="store_true", 
        help="Filter out examples that are both ambiguous (by cartography) AND non-questions (by rule-based detection).",
    )
    argp.add_argument(
        "--cartography_metrics_path",
        type=str,
        default=None,
        help="Path to existing cartography metrics. If not provided, filtering will be skipped.",
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

    # Fine-tuning head for QA task.
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ablations = args.ablations
    max_tokens_length = args.max_length
    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    prepare_train_dataset = partial(
        prepare_train_dataset_qa,
        tokenizer=tokenizer,
        ablations=ablations,
        max_seq_length=max_tokens_length,
    )
    prepare_eval_dataset = partial(prepare_validation_dataset_qa, tokenizer=tokenizer)
    print(
        "Preprocessing data... (this takes a little bit, should only happen once per dataset)"
    )

    # Dataset-specific preprocessing
    if dataset_name.lower() == "eladio/emrqa-msquad":
        # EMR-QA: Add ID column if missing
        for split in dataset.keys():
            if "id" not in dataset[split].column_names:
                dataset[split] = dataset[split].map(
                    generate_hash_ids, num_proc=NUM_PREPROCESSING_WORKERS
                )

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset["train"]

        # Filter ambiguous non-questions if requested and cartography metrics are available
        if args.filter_ambiguous_non_questions and args.cartography_metrics_path:
            print(f"\nFiltering ambiguous non-questions using cartography metrics from {args.cartography_metrics_path}")
            original_size = len(train_dataset)
            train_dataset = filter_ambiguous_non_questions(train_dataset, args.cartography_metrics_path)
            filtered_size = len(train_dataset)
            removed_count = original_size - filtered_size
            print(f"Removed {removed_count} ambiguous non-question examples ({removed_count/original_size*100:.1f}%)\n")
            
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

    # Initialize cartography callback if enabled
    cartography_callback = None
    if args.enable_cartography and training_args.do_train:
        cartography_callback = DatasetCartographyCallback(
            output_dir=args.cartography_output_dir
        )
        print(f"\n{'=' * 70}")
        print("DATASET CARTOGRAPHY ENABLED")
        print(f"{'=' * 70}")
        print(f"Output directory: {args.cartography_output_dir}")
        print("Training dynamics will be tracked across epochs.")
        print(f"{'=' * 70}\n")

    # TODO: Why are we doing this?
    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Create data collator that handles example_id for cartography
    data_collator = DataCollatorWithExampleId() if args.enable_cartography else None

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
        data_collator=data_collator,
        callbacks=[cartography_callback] if cartography_callback is not None else [],
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

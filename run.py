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
from dataset_filters import apply_filters
from helpers import (
    DataCollatorWithExampleId,
    QuestionAnsweringTrainer,
    compute_metrics,
    generate_hash_ids,
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
        help="Directory for cartography outputs. Used for: saving training dynamics (--enable_cartography), filtering (--filter_cartography), label smoothing (--use_label_smoothing), and soft weighting (--use_soft_weighting).",
    )
    argp.add_argument(
        "--filter_cartography",
        action="store_true",
        help="Filter dataset based on cartography metrics (removes ambiguous examples).",
    )
    argp.add_argument(
        "--filter_clusters",
        action="store_true",
        help="Filter dataset based on cluster assignments.",
    )
    argp.add_argument(
        "--cluster_assignments_path",
        type=str,
        default="./cluster_output",
        help="Path to cluster assignments directory or CSV file.",
    )
    argp.add_argument(
        "--exclude_clusters",
        type=str,
        default="-1",
        help="Comma-separated list of cluster IDs to exclude (e.g., '3,4,-1'). Default: '-1' (excludes noise). Use --exclude_clusters=\"-1,1\" when including negative numbers.",
    )

    argp.add_argument(
        "--min_cluster_probability",
        type=float,
        default=None,
        help="Minimum cluster probability threshold for filtering (0.0-1.0).",
    )
    argp.add_argument(
        "--use_label_smoothing",
        action="store_true",
        help="Enable variability-based label smoothing to reduce overfitting on ambiguous/noisy examples.",
    )
    argp.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.6,
        help="Multiplier to convert variability to smoothing amount (default: 0.6). Higher = more aggressive smoothing.",
    )
    argp.add_argument(
        "--use_soft_weighting",
        action="store_true",
        help="Enable soft weight schedule using variability from cartography metrics.",
    )
    argp.add_argument(
        "--weight_clip_min",
        type=float,
        default=0.1,
        help="Minimum weight value for soft weighting (default: 0.1).",
    )
    argp.add_argument(
        "--weight_clip_max",
        type=float,
        default=10.0,
        help="Maximum weight value for soft weighting (default: 10.0).",
    )
    argp.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use for training (default: 'train'). Use 'validation' to run cartography on validation data.",
    )
    argp.add_argument(
        "--filter_validation",
        action="store_true",
        help="Apply cartography filtering to validation set. Note: Only applies when --filter_cartography is enabled (not used with --filter_clusters).",
    )
    argp.add_argument(
        "--validation_cartography_output_dir",
        type=str,
        default="./cartography_output_validation",
        help="Directory with cartography metrics for validation set (if different from training). Default: ./cartography_output_validation",
    )
    argp.add_argument(
        "--validation_cluster_assignments_path",
        type=str,
        default="./cluster_output_validation",
        help="Path to cluster assignments for validation set (if different from training). Default: ./cluster_output_validation",
    )
    argp.add_argument(
        "--wandb_project",
        type=str,
        default="qa-cartography-experiments",
        help="Weights & Biases project name for experiment tracking.",
    )
    argp.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Custom run name for W&B (auto-generated if not specified).",
    )
    argp.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated tags for W&B run (e.g., 'baseline,filtered,smoothing').",
    )

    training_args, args = argp.parse_args_into_dataclasses()

    # Initialize Weights & Biases if enabled
    import wandb

    wandb_enabled = training_args.report_to and "wandb" in training_args.report_to

    if wandb_enabled:
        # Parse tags if provided
        tags = []
        if args.wandb_tags:
            tags = [tag.strip() for tag in args.wandb_tags.split(",")]

        # Auto-generate run name if not provided
        run_name = args.wandb_run_name
        if run_name is None:
            # Create descriptive name based on configuration
            name_parts = []
            if args.filter_cartography:
                name_parts.append("cart_filt")
            if args.filter_clusters:
                name_parts.append(f"clust_filt_excl{args.exclude_clusters}")
            if args.use_label_smoothing:
                name_parts.append(f"smooth{args.smoothing_factor}")
            if args.use_soft_weighting:
                name_parts.append(
                    f"weight_{args.weight_clip_min}-{args.weight_clip_max}"
                )

            run_name = "_".join(name_parts) if name_parts else "baseline"

        # Initialize wandb with configuration
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=tags,
            config={
                # Model and dataset
                "model": args.model,
                "dataset": args.dataset,
                "max_length": args.max_length,
                "ablations": args.ablations,
                # Training parameters (from TrainingArguments)
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "seed": training_args.seed,
                # Filtering strategies
                "filter_cartography": args.filter_cartography,
                "filter_clusters": args.filter_clusters,
                "exclude_clusters": args.exclude_clusters,
                "min_cluster_probability": args.min_cluster_probability,
                "filter_validation": args.filter_validation,
                # Training modifications
                "use_label_smoothing": args.use_label_smoothing,
                "smoothing_factor": args.smoothing_factor,
                "use_soft_weighting": args.use_soft_weighting,
                "weight_clip_min": args.weight_clip_min,
                "weight_clip_max": args.weight_clip_max,
                # Cartography
                "enable_cartography": args.enable_cartography,
            },
        )

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
        train_dataset = dataset[args.train_split]

        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))

        # Apply cartography filtering if requested
        if args.filter_cartography:
            print(f"\n{'=' * 70}")
            print("APPLYING CARTOGRAPHY FILTERING")
            print(f"{'=' * 70}")
            print(f"Using cartography metrics from: {args.cartography_output_dir}")

            original_size = len(train_dataset)

            # Create filter configuration
            #! Currently we're not using any of the rule-based.
            filter_config = {
                "ambiguous": {
                    "enabled": True,
                    "metrics_path": args.cartography_output_dir,
                    "top_fraction": 0.33,
                    "apply_rule_based_filter": False,
                }
            }

            # Apply filters
            train_dataset = apply_filters(train_dataset, filter_config)

            filtered_size = len(train_dataset)
            removed_count = original_size - filtered_size

            print("\nFiltering Summary:")
            print(f"  Original size: {original_size}")
            print(f"  Filtered size: {filtered_size}")
            print(
                f"  Removed: {removed_count} examples ({removed_count / original_size * 100:.1f}%)"
            )
            print(f"{'=' * 70}\n")

            # Log to wandb
            if wandb_enabled:
                wandb.log(
                    {
                        "train/original_size": original_size,
                        "train/filtered_size": filtered_size,
                        "train/removed_count": removed_count,
                        "train/removal_percentage": removed_count / original_size * 100,
                    }
                )

        # Apply cluster filtering if requested
        if args.filter_clusters:
            if args.cluster_assignments_path is None:
                print(
                    "\nWarning: --filter_clusters specified but no --cluster_assignments_path provided."
                )
                print("Skipping cluster filtering.\n")
            else:
                print(f"\n{'=' * 70}")
                print("APPLYING CLUSTER FILTERING")
                print(f"{'=' * 70}")
                print(
                    f"Using cluster assignments from: {args.cluster_assignments_path}"
                )

                original_size = len(train_dataset)

                # Parse exclude_clusters list (empty string means exclude nothing)
                exclude_clusters = []
                if args.exclude_clusters.strip():  # Only parse if not empty
                    exclude_clusters = [
                        int(c.strip()) for c in args.exclude_clusters.split(",")
                    ]

                # Create cluster filter configuration
                filter_config = {
                    "cluster": {
                        "enabled": True,
                        "cluster_path": args.cluster_assignments_path,
                        "exclude_clusters": exclude_clusters,
                        "min_probability": args.min_cluster_probability,
                    }
                }

                # Apply filters
                train_dataset = apply_filters(train_dataset, filter_config)

                filtered_size = len(train_dataset)
                removed_count = original_size - filtered_size

                print("\nCluster Filtering Summary:")
                print(f"  Original size: {original_size}")
                print(f"  Filtered size: {filtered_size}")
                print(
                    f"  Removed: {removed_count} examples ({removed_count / original_size * 100:.1f}%)"
                )
                print(f"{'=' * 70}\n")

                # Log to wandb
                if wandb_enabled:
                    wandb.log(
                        {
                            "train/cluster_original_size": original_size,
                            "train/cluster_filtered_size": filtered_size,
                            "train/cluster_removed_count": removed_count,
                            "train/cluster_removal_percentage": removed_count
                            / original_size
                            * 100,
                        }
                    )

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

        # Apply filtering to validation set if requested
        # Note: Only filter validation when using cartography filtering, not cluster filtering
        if args.filter_validation and args.filter_cartography:
            print(f"\n{'=' * 70}")
            print("APPLYING CARTOGRAPHY FILTERS TO VALIDATION SET")
            print(f"{'=' * 70}")

            original_eval_size = len(eval_dataset)

            # Determine paths for validation filtering
            val_cartography_dir = (
                args.validation_cartography_output_dir
                if args.validation_cartography_output_dir
                else args.cartography_output_dir
            )

            # Build filter config for validation (only cartography, not clusters)
            val_filter_config = {}

            val_filter_config["ambiguous"] = {
                "enabled": True,
                "metrics_path": val_cartography_dir,
                "top_fraction": 0.33,
                "apply_rule_based_filter": False,
            }
            print(f"  Cartography metrics from: {val_cartography_dir}")

            # Apply filters to validation set
            if val_filter_config:
                eval_dataset = apply_filters(eval_dataset, val_filter_config)

                filtered_eval_size = len(eval_dataset)
                removed_eval_count = original_eval_size - filtered_eval_size

                print("\nValidation Filtering Summary:")
                print(f"  Original size: {original_eval_size}")
                print(f"  Filtered size: {filtered_eval_size}")
                print(
                    f"  Removed: {removed_eval_count} examples ({removed_eval_count / original_eval_size * 100:.1f}%)"
                )

                # Log to wandb
                if wandb_enabled:
                    wandb.log(
                        {
                            "eval/original_size": original_eval_size,
                            "eval/filtered_size": filtered_eval_size,
                            "eval/removed_count": removed_eval_count,
                            "eval/removal_percentage": removed_eval_count
                            / original_eval_size
                            * 100,
                        }
                    )
            print(f"{'=' * 70}\n")

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

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    # Not the best way, creates hidden state decoupling between wrapper and the outer function.
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Create data collator that handles example_id for cartography, label smoothing, or soft weighting
    data_collator = (
        DataCollatorWithExampleId()
        if (
            args.enable_cartography
            or args.use_label_smoothing
            or args.use_soft_weighting
        )
        else None
    )

    # Load variability maps for label smoothing and/or soft weighting
    smoothing_map = None
    weighting_map = None

    if args.use_label_smoothing:
        print(f"\n{'=' * 70}")
        print("LABEL SMOOTHING ENABLED")
        print(f"{'=' * 70}")
        print(f"Loading variability metrics from: {args.cartography_output_dir}")

        from dataset_cartography import load_variability_map

        smoothing_map = load_variability_map(
            args.cartography_output_dir,
            mode="smoothing",
            smoothing_factor=args.smoothing_factor,
        )

        print(f"Loaded smoothing factors for {len(smoothing_map)} examples")

        # Print statistics
        smoothing_values = list(smoothing_map.values())
        print(
            f"Smoothing range: [{min(smoothing_values):.4f}, {max(smoothing_values):.4f}]"
        )
        print(f"Mean smoothing: {sum(smoothing_values) / len(smoothing_values):.4f}")

        # Show distribution
        low_smooth = sum(1 for s in smoothing_values if s < 0.05)
        med_smooth = sum(1 for s in smoothing_values if 0.05 <= s < 0.15)
        high_smooth = sum(1 for s in smoothing_values if s >= 0.15)

        print("\nSmoothing distribution:")
        print(
            f"  Low (<0.05):        {low_smooth:6d} ({100 * low_smooth / len(smoothing_values):5.1f}%)"
        )
        print(
            f"  Medium (0.05-0.15): {med_smooth:6d} ({100 * med_smooth / len(smoothing_values):5.1f}%)"
        )
        print(
            f"  High (≥0.15):       {high_smooth:6d} ({100 * high_smooth / len(smoothing_values):5.1f}%)"
        )
        print(f"{'=' * 70}\n")

    if args.use_soft_weighting:
        print(f"\n{'=' * 70}")
        print("SOFT WEIGHT SCHEDULE ENABLED")
        print(f"{'=' * 70}")
        print(f"Loading variability metrics from: {args.cartography_output_dir}")

        from dataset_cartography import load_variability_map

        weighting_map = load_variability_map(
            args.cartography_output_dir,
            mode="weighting",
            weight_clip_range=(args.weight_clip_min, args.weight_clip_max),
        )

        print(f"Loaded weights for {len(weighting_map)} examples")
        weights_list = list(weighting_map.values())
        print(f"Weight range: [{min(weights_list):.3f}, {max(weights_list):.3f}]")
        print(f"Mean weight: {sum(weights_list) / len(weights_list):.3f}")
        print(f"{'=' * 70}\n")

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        eval_examples=eval_dataset,  # Pass eval_examples for periodic evaluation
        processing_class=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
        data_collator=data_collator,
        smoothing_map=smoothing_map,
        weighting_map=weighting_map,
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

    # Finish wandb run
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()

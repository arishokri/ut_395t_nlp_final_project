#!/bin/bash
# Quick experiment runner with W&B integration
# Usage: ./run_experiment.sh <experiment_name> [additional args]

set -e

EXPERIMENT_NAME=${1:-"test_run"}
shift  # Remove first argument, keep the rest

# Default configuration
OUTPUT_DIR="./experiments/${EXPERIMENT_NAME}"
WANDB_PROJECT="qa-cartography-experiments"

echo "========================================================================"
echo "Running Experiment: ${EXPERIMENT_NAME}"
echo "========================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "W&B Project: ${WANDB_PROJECT}"
echo ""

# Run training
python3 run.py \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${EXPERIMENT_NAME}" \
  --report_to wandb \
  --do_train \
  --do_eval \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 100 \
  --save_total_limit 2 \
  --load_best_model_at_end true \
  --metric_for_best_model f1 \
  "$@"

echo ""
echo "========================================================================"
echo "Experiment completed: ${EXPERIMENT_NAME}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================================================"

#!/bin/bash
# Interactive Training Script
# Trains and evaluates a model based on parameters in train_config.yaml
# User only needs to specify run name and dataset sizes interactively

set -e

# Color codes for better UX
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function to ask yes/no questions
ask_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    
    if [ "$default" = "y" ]; then
        read -p "$prompt (Y/n) " -n 1 -r < /dev/tty
    else
        read -p "$prompt (y/N) " -n 1 -r < /dev/tty
    fi
    echo
    
    if [ -z "$REPLY" ]; then
        [[ "$default" = "y" ]]
    else
        [[ $REPLY =~ ^[Yy]$ ]]
    fi
}

# Helper function to get numeric input with default
ask_number() {
    local prompt="$1"
    local default="$2"
    
    read -p "$prompt [default: $default]: " input < /dev/tty
    
    # Use default if empty
    if [ -z "$input" ]; then
        echo "$default"
    else
        echo "$input"
    fi
}

# Helper function to get text input with default
ask_text() {
    local prompt="$1"
    local default="$2"
    
    read -p "$prompt [default: $default]: " input < /dev/tty
    
    # Use default if empty
    if [ -z "$input" ]; then
        echo "$default"
    else
        echo "$input"
    fi
}

# =============================================================================
# Welcome and Configuration Source Selection
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Interactive Training Script"
echo -e "========================================================================${NC}"
echo ""
echo "This script will train and evaluate a model based on your configuration."
echo ""

# Ask user which config to use
echo -e "${CYAN}Step 1: Configuration Source${NC}"
echo ""
echo "Choose configuration source:"
echo "  1) Custom config file (train_config_custom.yaml)"
echo "  2) W&B sweep config file (train_config_wandb.yaml)"
echo ""

while true; do
    read -p "Enter choice [1 or 2]: " -n 1 -r choice < /dev/tty
    echo
    
    if [[ "$choice" == "1" ]]; then
        CONFIG_FILE="train_config_custom.yaml"
        CONFIG_TYPE="custom"
        break
    elif [[ "$choice" == "2" ]]; then
        CONFIG_FILE="train_config_wandb.yaml"
        CONFIG_TYPE="wandb"
        break
    else
        echo -e "${RED}Invalid choice. Please enter 1 or 2.${NC}"
    fi
done

echo ""
echo -e "${CYAN}Using configuration file: ${CONFIG_FILE}${NC}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found!${NC}"
    echo ""
    if [ "$CONFIG_TYPE" == "custom" ]; then
        echo "Please create a train_config_custom.yaml file with your training parameters."
        echo "You can use the existing file as a template."
    else
        echo "Please create a train_config_wandb.yaml file with your W&B sweep config."
        echo "You can download config.yaml from a W&B run and rename it:"
        echo "  mv config.yaml train_config_wandb.yaml"
    fi
    echo ""
    exit 1
fi

# Check for Python and required packages
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Verify PyYAML is installed
if ! python3 -c "import yaml" &> /dev/null; then
    echo -e "${YELLOW}Installing PyYAML...${NC}"
    pip install pyyaml
fi

# =============================================================================
# Get Run Name and Dataset Sizes
# =============================================================================

# Get run name
echo -e "${CYAN}Step 2: Run Configuration${NC}"
RUN_NAME=$(ask_text "Enter a name for this training run" "baseline")

# Get dataset sizes
echo ""
echo -e "${CYAN}Step 3: Dataset Sizes${NC}"
# Get dataset sizes
echo ""
echo -e "${CYAN}Step 3: Dataset Sizes${NC}"
echo "Enter 0 to use the full dataset"
MAX_TRAIN=$(ask_number "Maximum training samples" "0")
MAX_EVAL=$(ask_number "Maximum evaluation samples" "0")

# Set output directory
MODEL_DIR="./trained_models/train_${RUN_NAME}"

# Create directory
mkdir -p "$MODEL_DIR"

# Check if directories already exist with content
if [ -n "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo ""
    echo -e "${YELLOW}Warning: Output directory '$MODEL_DIR' is not empty!${NC}"
    if ! ask_yes_no "Continue and overwrite?" "n"; then
        echo "Exiting..."
        exit 0
    fi
fi

# =============================================================================
# Load and Display Configuration
# =============================================================================

echo ""
echo -e "${BLUE}========================================================================"
echo "Loading Configuration from $CONFIG_FILE"
echo -e "========================================================================${NC}"

# Parse YAML config using Python
read -r -d '' PYTHON_SCRIPT << 'EOF' || true
import yaml
import sys
import re

with open(sys.argv[1], 'r') as f:
    config = yaml.safe_load(f)

config_type = sys.argv[2]

# For W&B configs, extract the 'value' field from each parameter
if config_type == 'wandb':
    config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v 
              for k, v in config.items()}

# Print configuration, filtering out keys with invalid bash variable characters
for key, value in config.items():
    # Skip keys that contain invalid bash variable name characters (/, ., etc.)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
        continue
    
    if value is None:
        print(f"{key}=null")
    elif isinstance(value, bool):
        print(f"{key}={'true' if value else 'false'}")
    else:
        print(f"{key}={value}")
EOF

# Load config into bash variables
while IFS='=' read -r key value; do
    export "CONFIG_$key"="$value"
done < <(python3 -c "$PYTHON_SCRIPT" "$CONFIG_FILE" "$CONFIG_TYPE")

# Display key configuration
echo ""
echo -e "${CYAN}Run Configuration:${NC}"
echo "  Run name: $RUN_NAME"
echo "  Model: $CONFIG_model"
echo "  Dataset: $CONFIG_dataset"
echo "  Training samples: $([ $MAX_TRAIN -eq 0 ] && echo 'full dataset' || echo $MAX_TRAIN)"
echo "  Evaluation samples: $([ $MAX_EVAL -eq 0 ] && echo 'full dataset' || echo $MAX_EVAL)"
echo "  Epochs: $CONFIG_num_train_epochs"
echo "  Batch size: $CONFIG_per_device_train_batch_size"
echo "  Learning rate: $CONFIG_learning_rate"
echo "  Seed: $CONFIG_seed"
echo "  Max length: $CONFIG_max_length"
echo ""
echo -e "${CYAN}Filtering & Modifications:${NC}"
echo "  Cartography: $CONFIG_enable_cartography"
echo "  Filter ambiguous: $CONFIG_filter_ambiguous"
if [ "$CONFIG_filter_ambiguous" = "true" ]; then
    echo "    ├─ Top fraction: $CONFIG_ambiguous_top_fraction"
    echo "    └─ Variability margin: $CONFIG_variability_margin"
fi
echo "  Filter clusters: $CONFIG_filter_clusters"
if [ "$CONFIG_filter_clusters" = "true" ]; then
    echo "    ├─ Min probability: $CONFIG_min_cluster_probability"
    echo "    └─ Exclude noise: $CONFIG_exclude_noise_cluster"
fi
echo "  Filter rule-based: $CONFIG_filter_rule_based"
if [ "$CONFIG_filter_rule_based" = "true" ]; then
    echo "    ├─ Rule name: $CONFIG_rule_name"
    echo "    └─ Similarity threshold: $CONFIG_rule_sim_threshold"
fi
echo "  Filter validation: $CONFIG_filter_validation"
echo "  Label smoothing: $CONFIG_use_label_smoothing"
if [ "$CONFIG_use_label_smoothing" = "true" ]; then
    echo "    └─ Smoothing factor: $CONFIG_smoothing_factor"
fi
echo "  Soft weighting: $CONFIG_use_soft_weighting"
if [ "$CONFIG_use_soft_weighting" = "true" ]; then
    echo "    ├─ Weight clip min: $CONFIG_weight_clip_min"
    echo "    └─ Weight clip max: $CONFIG_weight_clip_max"
fi
echo ""
echo -e "${CYAN}Output Directory:${NC}"
echo "  Model: $MODEL_DIR"
echo ""

# Confirm before starting
if ! ask_yes_no "Start training with this configuration?" "y"; then
    echo "Exiting..."
    exit 0
fi

# =============================================================================
# Build Command
# =============================================================================

echo ""
echo -e "${BLUE}========================================================================"
echo "Building Training Command"
echo -e "========================================================================${NC}"

# Build command arguments
CMD_ARGS=(
    "--model" "$CONFIG_model"
    "--dataset" "$CONFIG_dataset"
    "--max_length" "$CONFIG_max_length"
    "--do_train"
    "--do_eval"
    "--num_train_epochs" "$CONFIG_num_train_epochs"
    "--per_device_train_batch_size" "$CONFIG_per_device_train_batch_size"
    "--learning_rate" "$CONFIG_learning_rate"
    "--seed" "$CONFIG_seed"
    "--output_dir" "$MODEL_DIR"
)

# Handle eval and save strategy
# If load_best_model_at_end is true, strategies must match
if [ "$CONFIG_load_best_model_at_end" = "true" ]; then
    # Use eval_strategy for both to ensure they match
    CMD_ARGS+=("--eval_strategy" "$CONFIG_eval_strategy")
    CMD_ARGS+=("--save_strategy" "$CONFIG_eval_strategy")
else
    CMD_ARGS+=("--eval_strategy" "$CONFIG_eval_strategy")
    CMD_ARGS+=("--save_strategy" "$CONFIG_save_strategy")
fi

CMD_ARGS+=(
    "--save_total_limit" "$CONFIG_save_total_limit"
    "--logging_steps" "$CONFIG_logging_steps"
    "--report_to" "wandb"
    "--wandb_project" "$CONFIG_wandb_project"
    "--wandb_run_name" "$RUN_NAME"
)

# Add max samples if specified
if [ $MAX_TRAIN -ne 0 ]; then
    CMD_ARGS+=("--max_train_samples" "$MAX_TRAIN")
fi

if [ $MAX_EVAL -ne 0 ]; then
    CMD_ARGS+=("--max_eval_samples" "$MAX_EVAL")
fi

# Add optional boolean flags
if [ "$CONFIG_enable_cartography" = "true" ]; then
    CMD_ARGS+=("--enable_cartography")
    CMD_ARGS+=("--cartography_output_dir" "$CONFIG_cartography_output_dir")
fi

if [ "$CONFIG_filter_ambiguous" = "true" ]; then
    CMD_ARGS+=("--filter_ambiguous")
    CMD_ARGS+=("--ambiguous_top_fraction" "$CONFIG_ambiguous_top_fraction")
    CMD_ARGS+=("--variability_margin" "$CONFIG_variability_margin")
    if [ "$CONFIG_enable_cartography" = "false" ]; then
        CMD_ARGS+=("--cartography_output_dir" "$CONFIG_cartography_output_dir")
    fi
fi

if [ "$CONFIG_filter_clusters" = "true" ]; then
    CMD_ARGS+=("--filter_clusters")
    CMD_ARGS+=("--cluster_assignments_path" "$CONFIG_cluster_assignments_path")
    if [ "$CONFIG_exclude_noise_cluster" = "true" ]; then
        CMD_ARGS+=("--exclude_noise_cluster")
    fi
    if [ "$CONFIG_min_cluster_probability" != "null" ] && [ -n "$CONFIG_min_cluster_probability" ]; then
        CMD_ARGS+=("--min_cluster_probability" "$CONFIG_min_cluster_probability")
    fi
fi

if [ "$CONFIG_filter_rule_based" = "true" ]; then
    CMD_ARGS+=("--filter_rule_based")
    CMD_ARGS+=("--rule_name" "$CONFIG_rule_name")
    CMD_ARGS+=("--rule_sim_threshold" "$CONFIG_rule_sim_threshold")
fi

if [ "$CONFIG_filter_validation" = "true" ]; then
    CMD_ARGS+=("--filter_validation")
    if [ "$CONFIG_filter_clusters" = "true" ]; then
        CMD_ARGS+=("--validation_cluster_assignments_path" "$CONFIG_validation_cluster_assignments_path")
    fi
fi

if [ "$CONFIG_use_label_smoothing" = "true" ]; then
    CMD_ARGS+=("--use_label_smoothing")
    CMD_ARGS+=("--smoothing_factor" "$CONFIG_smoothing_factor")
fi

if [ "$CONFIG_use_soft_weighting" = "true" ]; then
    CMD_ARGS+=("--use_soft_weighting")
    CMD_ARGS+=("--weight_clip_min" "$CONFIG_weight_clip_min")
    CMD_ARGS+=("--weight_clip_max" "$CONFIG_weight_clip_max")
fi

if [ "$CONFIG_load_best_model_at_end" = "true" ]; then
    CMD_ARGS+=("--load_best_model_at_end")
    CMD_ARGS+=("--metric_for_best_model" "$CONFIG_metric_for_best_model")
fi

# Add W&B tags if specified
if [ -n "$CONFIG_wandb_tags" ] && [ "$CONFIG_wandb_tags" != "" ]; then
    CMD_ARGS+=("--wandb_tags" "$CONFIG_wandb_tags")
fi

# =============================================================================
# Run Training
# =============================================================================

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo ""

START_TIME=$(date +%s)

# Run the training command and save output
LOG_FILE="$MODEL_DIR/training.log"

if python3 run.py "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}========================================================================"
    echo "Training Completed Successfully!"
    echo -e "========================================================================${NC}"
    echo "Duration: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
    echo ""
    
    # Check if evaluation metrics exist
    if [ -f "$MODEL_DIR/eval_metrics.json" ]; then
        echo -e "${CYAN}Evaluation Results:${NC}"
        python3 -c "
import json
with open('$MODEL_DIR/eval_metrics.json', 'r') as f:
    metrics = json.load(f)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f'  {key}: {value:.4f}')
        else:
            print(f'  {key}: {value}')
"
        echo ""
    fi
    
    echo ""
    echo -e "${CYAN}Output Location:${NC}"
    echo "  $MODEL_DIR"
    echo "    ├─ Model checkpoint (pytorch_model.bin, config.json)"
    echo "    ├─ Evaluation results (eval_metrics.json, eval_predictions.jsonl)"
    echo "    ├─ Training log (training.log)"
    echo "    └─ Checkpoints (checkpoint-*/ directories)"
    echo ""
    
    # Cleanup option
    echo -e "${YELLOW}Cleanup:${NC}"
    
    # Check if there are checkpoint directories
    if ls -d "$MODEL_DIR"/checkpoint-* &>/dev/null; then
        if ask_yes_no "Remove intermediate checkpoint directories? (keeps final model)" "n"; then
            rm -rf "$MODEL_DIR"/checkpoint-*
            echo -e "${GREEN}✓ Removed checkpoint directories${NC}"
        fi
    fi
    
else
    echo ""
    echo -e "${RED}========================================================================"
    echo "Training Failed!"
    echo -e "========================================================================${NC}"
    echo "Check the log file for details: $LOG_FILE"
    echo ""
    exit 1
fi

echo ""
echo -e "${BLUE}========================================================================${NC}"
echo ""

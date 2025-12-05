#!/bin/bash
# Interactive Ablation Study Runner
# Compares baseline (none) vs question-only (q_only) vs passage-only (p_only) ablations
# Runs multiple seeds and generates statistical analysis with visualizations

set -e

# Default configuration
DATASET="Eladio/emrqa-msquad"
MODEL="google/electra-small-discriminator"
NUM_EPOCHS=3
MAX_LENGTH=512
WANDB_PROJECT="qa-ablation-study"

# Color codes for better UX
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
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
    local min="$3"
    local max="$4"
    
    while true; do
        read -p "$prompt [default: $default]: " input < /dev/tty
        
        # Use default if empty
        if [ -z "$input" ]; then
            echo "$default"
            return
        fi
        
        # Check if numeric
        if ! [[ "$input" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}Please enter a valid number${NC}" >&2
            continue
        fi
        
        # Check range if specified
        if [ -n "$min" ] && [ "$input" -lt "$min" ]; then
            echo -e "${RED}Value must be at least $min${NC}" >&2
            continue
        fi
        
        if [ -n "$max" ] && [ "$input" -gt "$max" ]; then
            echo -e "${RED}Value must be at most $max${NC}" >&2
            continue
        fi
        
        echo "$input"
        return
    done
}

# =============================================================================
# Welcome and Configuration
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Ablation Study Runner"
echo -e "========================================================================${NC}"
echo ""
echo "This script compares model performance across ablation conditions:"
echo -e "  • ${CYAN}none${NC}    - Full model (question + passage)"
echo -e "  • ${CYAN}q_only${NC}  - Question-only (passage masked out)"
echo -e "  • ${CYAN}p_only${NC}  - Passage-only (question replaced with generic prompt)"
echo ""
echo "The script will:"
echo "  1. Run training & evaluation for each ablation across multiple seeds"
echo "  2. Compute statistics (mean, std, confidence intervals)"
echo "  3. Perform statistical significance tests (t-tests, effect sizes)"
echo "  4. Generate comparison visualizations"
echo ""
echo -e "${BLUE}========================================================================${NC}"
echo ""

# =============================================================================
# Configuration Parameters
# =============================================================================

echo -e "${CYAN}Configuration${NC}"
echo "------------------------------------------------------------------------"
echo ""

# Number of seeds
NUM_SEEDS=$(ask_number "Number of random seeds to run (1-5)" "3" "1" "5")
echo -e "${GREEN}✓${NC} Will run ${NUM_SEEDS} seed(s) per ablation"
echo ""

# Batch size
BATCH_SIZE=$(ask_number "Per-device batch size" "64" "1" "128")
echo -e "${GREEN}✓${NC} Batch size: ${BATCH_SIZE}"
echo ""

# Max train samples
MAX_TRAIN=$(ask_number "Maximum training samples (0 = use all)" "5000" "0" "")
if [ "$MAX_TRAIN" -eq "0" ]; then
    MAX_TRAIN_ARG=""
    echo -e "${GREEN}✓${NC} Using all training samples"
else
    MAX_TRAIN_ARG="--max_train_samples ${MAX_TRAIN}"
    echo -e "${GREEN}✓${NC} Training samples: ${MAX_TRAIN}"
fi
echo ""

# Max eval samples
MAX_EVAL=$(ask_number "Maximum evaluation samples (0 = use all)" "1000" "0" "")
if [ "$MAX_EVAL" -eq "0" ]; then
    MAX_EVAL_ARG=""
    echo -e "${GREEN}✓${NC} Using all evaluation samples"
else
    MAX_EVAL_ARG="--max_eval_samples ${MAX_EVAL}"
    echo -e "${GREEN}✓${NC} Evaluation samples: ${MAX_EVAL}"
fi
echo ""

# =============================================================================
# Check for Existing Baseline Models
# =============================================================================

echo -e "${CYAN}Baseline Model Selection${NC}"
echo "------------------------------------------------------------------------"
echo ""

# Only allow model reuse if running on full dataset (no sample limits)
ALLOW_MODEL_REUSE=false
if [ "$MAX_TRAIN" -eq "0" ] && [ "$MAX_EVAL" -eq "0" ]; then
    ALLOW_MODEL_REUSE=true
    echo "Running on full dataset - baseline model reuse available"
    echo ""
else
    echo "Running with limited samples - will train all models from scratch"
    echo -e "${YELLOW}Note: Model reuse only available when running on full dataset${NC}"
    echo ""
fi

BASELINE_MODEL_PATH="${MODEL}"
REUSE_BASELINE=false

# Check for fine-tuned model only if reuse is allowed
if [ "$ALLOW_MODEL_REUSE" = true ]; then
    DEFAULT_TRAINED_MODEL="./trained_models"
    
    # Check for default trained model
    if [ -d "$DEFAULT_TRAINED_MODEL" ] && [ -f "$DEFAULT_TRAINED_MODEL/config.json" ]; then
        echo -e "${GREEN}✓${NC} Found fine-tuned model: ${DEFAULT_TRAINED_MODEL}"
        echo "  This model can be used as the starting point for baseline (none) ablation"
        echo ""
        
        echo "Baseline model options:"
        echo "  [d] Use fine-tuned model (${DEFAULT_TRAINED_MODEL})"
        echo "  [c] Specify custom model directory"
        echo "  [0] Train new baseline from pretrained model"
        echo ""
        
        while true; do
            read -p "Select option: " choice
            
            # Handle default trained model
            if [ "$choice" = "d" ] || [ "$choice" = "D" ]; then
                BASELINE_MODEL_PATH="$DEFAULT_TRAINED_MODEL"
                REUSE_BASELINE=true
                echo -e "${GREEN}✓${NC} Using fine-tuned model: ${DEFAULT_TRAINED_MODEL}"
                echo -e "${CYAN}Note: This will be used only for baseline (none) ablation${NC}"
                echo -e "${CYAN}      Ablated models (q_only, p_only) will train from scratch${NC}"
                break
            fi
            
            # Handle custom directory
            if [ "$choice" = "c" ] || [ "$choice" = "C" ]; then
                while true; do
                    read -p "Enter model directory path: " custom_path
                    
                    if [ -z "$custom_path" ]; then
                        echo -e "${RED}Path cannot be empty${NC}"
                        continue
                    fi
                    
                    if [ ! -d "$custom_path" ]; then
                        echo -e "${RED}Directory not found: ${custom_path}${NC}"
                        if ! ask_yes_no "Try another path?" "y"; then
                            break
                        fi
                        continue
                    fi
                    
                    if [ ! -f "$custom_path/config.json" ]; then
                        echo -e "${YELLOW}Warning: No config.json found in ${custom_path}${NC}"
                        if ! ask_yes_no "Use this directory anyway?" "n"; then
                            if ! ask_yes_no "Try another path?" "y"; then
                                break
                            fi
                            continue
                        fi
                    fi
                    
                    BASELINE_MODEL_PATH="$custom_path"
                    REUSE_BASELINE=true
                    echo -e "${GREEN}✓${NC} Using custom model: ${custom_path}"
                    echo -e "${CYAN}Note: This will be used only for baseline (none) ablation${NC}"
                    echo -e "${CYAN}      Ablated models (q_only, p_only) will train from scratch${NC}"
                    break
                done
                
                if [ "$REUSE_BASELINE" = true ]; then
                    break
                fi
                
                # If user cancelled custom path, go back to main menu
                continue
            fi
            
            # Handle train new
            if [ "$choice" = "0" ]; then
                echo -e "${GREEN}✓${NC} Will train new baseline from pretrained model"
                break
            fi
            
            echo -e "${RED}Invalid selection${NC}"
        done
    else
        echo "No fine-tuned model found at ${DEFAULT_TRAINED_MODEL}"
        echo -e "${GREEN}✓${NC} Will train all models from pretrained model"
    fi
fi

echo ""

# =============================================================================
# Experiment Planning
# =============================================================================

ABLATIONS=("none" "q_only" "p_only")
TOTAL_EXPERIMENTS=$((${#ABLATIONS[@]} * NUM_SEEDS))

echo -e "${BLUE}========================================================================"
echo "Experiment Plan"
echo -e "========================================================================${NC}"
echo ""
echo "Ablations:       ${ABLATIONS[@]}"
echo "Seeds per run:   ${NUM_SEEDS}"
echo "Total runs:      ${TOTAL_EXPERIMENTS}"
echo "Epochs:          ${NUM_EPOCHS}"
echo "Dataset:         ${DATASET}"
echo "Model:           ${MODEL}"
echo ""
echo "Output locations:"
echo "  Experiments:   ./experiments/ablation_<type>_seed<N>/"
echo "  Logs:          ./experiments/ablation_<type>_seed<N>.log"
echo "  Analysis:      ./ablation_results/"
echo ""

if ! ask_yes_no "Proceed with ablation study?" "y"; then
    echo -e "${YELLOW}Aborted by user.${NC}"
    exit 0
fi

echo ""

# =============================================================================
# Run Experiments
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Running Experiments"
echo -e "========================================================================${NC}"
echo ""

# Track start time
START_TIME=$(date +%s)
COMPLETED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

# Arrays to track experiment results
declare -a EXPERIMENT_DIRS
declare -a EXPERIMENT_STATUS

for ablation in "${ABLATIONS[@]}"; do
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}Ablation: ${ablation}${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    for seed in $(seq 1 $NUM_SEEDS); do
        # Generate seed value (42, 43, 44, etc.)
        SEED_VALUE=$((41 + seed))
        
        EXP_NAME="ablation_${ablation}_seed${SEED_VALUE}"
        EXP_DIR="./experiments/${EXP_NAME}"
        
        echo -e "${CYAN}[$((COMPLETED_COUNT + SKIPPED_COUNT + 1))/${TOTAL_EXPERIMENTS}] ${EXP_NAME}${NC}"
        
        # Check if experiment already exists
        if [ -f "$EXP_DIR/eval_metrics.json" ]; then
            echo -e "${YELLOW}  ⚠ Experiment results already exist${NC}"
            
            # Show existing results
            f1=$(python3 -c "import json; print(f\"{json.load(open('$EXP_DIR/eval_metrics.json')).get('eval_f1', 0):.4f}\")" 2>/dev/null || echo "N/A")
            em=$(python3 -c "import json; print(f\"{json.load(open('$EXP_DIR/eval_metrics.json')).get('eval_exact_match', 0):.4f}\")" 2>/dev/null || echo "N/A")
            echo -e "    Existing results: F1=${f1}, EM=${em}"
            
            if ask_yes_no "  Reuse existing results?" "y"; then
                echo -e "${GREEN}  ✓ Reusing existing results${NC}"
                EXPERIMENT_DIRS+=("$EXP_DIR")
                EXPERIMENT_STATUS+=("reused")
                SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                echo ""
                continue
            else
                echo -e "${YELLOW}  ⟳ Retraining from scratch...${NC}"
                # Remove old results to ensure fresh eval
                rm -f "$EXP_DIR/eval_metrics.json" "$EXP_DIR/eval_predictions.jsonl"
            fi
        fi
        
        # Determine model path and training mode
        # Only use fine-tuned model for baseline (none) ablation when reuse is enabled
        # Ablated models (q_only, p_only) always train from scratch
        DO_TRAIN_ARG="--do_train"
        
        if [ "$ablation" = "none" ] && [ "$REUSE_BASELINE" = true ]; then
            MODEL_ARG="--model ${BASELINE_MODEL_PATH}"
            DO_TRAIN_ARG=""  # Skip training, only evaluate
            echo -e "  ${CYAN}Using fine-tuned model (eval only):${NC} $(basename "$BASELINE_MODEL_PATH")"
        else
            MODEL_ARG="--model ${MODEL}"
            if [ "$ablation" != "none" ]; then
                echo -e "  ${CYAN}Training ablated model from scratch${NC}"
            fi
        fi
        
        # Run experiment
        if [ -z "$DO_TRAIN_ARG" ]; then
            echo -e "  Running evaluation..."
        else
            echo -e "  Running training..."
        fi
        
        EXP_START=$(date +%s)
        
        if python3 run.py \
            $MODEL_ARG \
            --dataset "${DATASET}" \
            --max_length "${MAX_LENGTH}" \
            --ablations "${ablation}" \
            $DO_TRAIN_ARG \
            --do_eval \
            --num_train_epochs "${NUM_EPOCHS}" \
            --per_device_train_batch_size "${BATCH_SIZE}" \
            $MAX_TRAIN_ARG \
            $MAX_EVAL_ARG \
            --seed "${SEED_VALUE}" \
            --output_dir "${EXP_DIR}" \
            --eval_strategy epoch \
            --save_strategy epoch \
            --save_total_limit 1 \
            --logging_steps 50 \
            --report_to wandb \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_run_name "${EXP_NAME}" \
            --wandb_tags "ablation,${ablation},seed_${SEED_VALUE}" \
            2>&1 | tee "${EXP_DIR}.log"; then
            
            EXP_END=$(date +%s)
            EXP_DURATION=$((EXP_END - EXP_START))
            
            # Read results
            if [ -f "$EXP_DIR/eval_metrics.json" ]; then
                f1=$(python3 -c "import json; print(f\"{json.load(open('$EXP_DIR/eval_metrics.json')).get('eval_f1', 0):.4f}\")")
                em=$(python3 -c "import json; print(f\"{json.load(open('$EXP_DIR/eval_metrics.json')).get('eval_exact_match', 0):.4f}\")")
                
                echo -e "${GREEN}  ✓ Completed in ${EXP_DURATION}s${NC}"
                echo -e "    Results: F1=${f1}, EM=${em}"
                
                EXPERIMENT_DIRS+=("$EXP_DIR")
                EXPERIMENT_STATUS+=("completed")
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            else
                echo -e "${RED}  ✗ No results file generated${NC}"
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi
        else
            echo -e "${RED}  ✗ Training failed (see ${EXP_DIR}.log)${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        
        echo ""
    done
    
    echo ""
done

# =============================================================================
# Analysis
# =============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo -e "${BLUE}========================================================================"
echo "Analysis"
echo -e "========================================================================${NC}"
echo ""
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Completed: ${COMPLETED_COUNT}, Reused: ${SKIPPED_COUNT}, Failed: ${FAILED_COUNT}"
echo ""

if [ $((COMPLETED_COUNT + SKIPPED_COUNT)) -lt 3 ]; then
    echo -e "${RED}Insufficient experiments completed for analysis.${NC}"
    echo "Need at least one experiment per ablation type (3 total)."
    exit 1
fi

echo -e "${CYAN}Running statistical analysis...${NC}"
echo ""

# Create output directory for analysis
ANALYSIS_DIR="./ablation_results"
mkdir -p "$ANALYSIS_DIR"

# Run analysis script
if python3 analyze_ablations.py --experiment_dir ./experiments --output_dir "$ANALYSIS_DIR"; then
    echo ""
    echo -e "${GREEN}✓ Analysis complete!${NC}"
    echo ""
    
    # Display results summary
    if [ -f "$ANALYSIS_DIR/summary.txt" ]; then
        cat "$ANALYSIS_DIR/summary.txt"
    fi
else
    echo -e "${RED}✗ Analysis failed${NC}"
    exit 1
fi

# =============================================================================
# Summary and Next Steps
# =============================================================================

echo ""
echo -e "${BLUE}========================================================================"
echo "Ablation Study Complete! 🎉"
echo -e "========================================================================${NC}"
echo ""
echo -e "${GREEN}Generated outputs:${NC}"
echo "  • Summary statistics:    ${ANALYSIS_DIR}/summary.csv"
echo "  • Detailed results:      ${ANALYSIS_DIR}/detailed_results.json"
echo "  • Comparison plot:       ${ANALYSIS_DIR}/ablation_comparison.png"
echo "  • Delta plot:            ${ANALYSIS_DIR}/ablation_deltas.png"
echo "  • Scatter plot:          ${ANALYSIS_DIR}/seed_scatter.png"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. View visualizations:"
echo "     xdg-open ${ANALYSIS_DIR}/ablation_comparison.png"
echo ""
echo "  2. Review detailed results:"
echo "     cat ${ANALYSIS_DIR}/summary.csv"
echo "     cat ${ANALYSIS_DIR}/detailed_results.json"
echo ""
echo "  3. Check experiment logs:"
echo "     ls -lh ./experiments/ablation_*.log"
echo ""

# Cleanup option
if [ $((COMPLETED_COUNT + SKIPPED_COUNT)) -gt 0 ]; then
    echo -e "${YELLOW}Cleanup options:${NC}"
    echo "  Model checkpoints are stored in ./experiments/ablation_*/"
    echo ""
    
    if ask_yes_no "Remove model checkpoints to save space (keep metrics)?" "n"; then
        echo ""
        for exp_dir in "${EXPERIMENT_DIRS[@]}"; do
            if [ -d "$exp_dir" ]; then
                # Keep only eval_metrics.json and eval_predictions.jsonl
                find "$exp_dir" -type f ! -name "eval_metrics.json" ! -name "eval_predictions.jsonl" -delete 2>/dev/null
                find "$exp_dir" -type d -empty -delete 2>/dev/null
            fi
        done
        echo -e "${GREEN}✓ Cleaned up model checkpoints${NC}"
    fi
fi

echo ""
echo -e "${BLUE}========================================================================${NC}"

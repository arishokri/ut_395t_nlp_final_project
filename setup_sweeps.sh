#!/bin/bash
# Complete setup script for hyperparameter search system
# This script handles both initial setup and data preparation

set -e

DATASET="Eladio/emrqa-msquad"
MODEL="google/electra-small-discriminator"
NUM_EPOCHS=5
MAX_LENGTH=512
PER_DEVICE_BATCHES=64

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
        read -p "$prompt (Y/n) " -n 1 -r
    else
        read -p "$prompt (y/N) " -n 1 -r
    fi
    echo
    
    if [ -z "$REPLY" ]; then
        [[ "$default" = "y" ]]
    else
        [[ $REPLY =~ ^[Yy]$ ]]
    fi
}

# Helper function to check if file exists and show status
check_file_status() {
    local file_path="$1"
    local description="$2"
    
    if [ -f "$file_path" ] || [ -d "$file_path" ]; then
        echo -e "${GREEN}✓${NC} $description"
        echo -e "  ${CYAN}Location:${NC} $file_path"
        return 0
    else
        echo -e "${YELLOW}○${NC} $description"
        echo -e "  ${CYAN}Expected location:${NC} $file_path"
        return 1
    fi
}

# =============================================================================
# Initial Setup: Dependencies and W&B Login
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Hyperparameter Search System - Interactive Setup"
echo -e "========================================================================${NC}"
echo ""
echo "This script will guide you through setting up the data preparation"
echo "required for running hyperparameter sweeps."
echo ""

# Check if wandb is installed
echo -e "${CYAN}Step 1: Checking dependencies...${NC}"
if ! python3 -c "import wandb" 2>/dev/null; then
    echo -e "${YELLOW}wandb is not installed.${NC}"
    if ask_yes_no "Install dependencies now?" "y"; then
        uv sync
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    else
        echo -e "${RED}Cannot proceed without dependencies. Exiting.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

echo ""

# Check if logged into wandb
echo -e "${CYAN}Step 2: Checking W&B authentication...${NC}"
if ! wandb login --check 2>/dev/null; then
    echo -e "${YELLOW}Not logged in to Weights & Biases.${NC}"
    if ask_yes_no "Login to W&B now?" "y"; then
        wandb login
        echo -e "${GREEN}✓ Logged in to W&B${NC}"
    else
        echo -e "${YELLOW}Warning: W&B experiments will not be tracked.${NC}"
    fi
else
    echo -e "${GREEN}✓ Already logged in to W&B${NC}"
fi

echo ""
echo -e "${BLUE}========================================================================"
echo "Data Preparation Checklist"
echo -e "========================================================================${NC}"
echo ""
echo "The following data files are needed for experiments:"
echo ""

echo "The following data files are needed for experiments:"
echo ""

# Check status of all required files
TRAIN_CART_STATUS=1
VAL_CART_STATUS=1
TRAIN_CLUSTER_STATUS=1
VAL_CLUSTER_STATUS=1

echo -e "${YELLOW}Required for all experiments:${NC}"
check_file_status "./cartography_output/cartography_metrics.csv" "Training cartography metrics" && TRAIN_CART_STATUS=0
check_file_status "./cluster_output/cluster_assignments.csv" "Training cluster assignments" && TRAIN_CLUSTER_STATUS=0

echo ""
echo -e "${YELLOW}Optional (for validation filtering):${NC}"
check_file_status "./cartography_output_validation/cartography_metrics.csv" "Validation cartography metrics" && VAL_CART_STATUS=0
check_file_status "./cluster_output_validation/cluster_assignments.csv" "Validation cluster assignments" && VAL_CLUSTER_STATUS=0

echo ""
echo -e "${BLUE}========================================================================${NC}"
echo ""

if [ $TRAIN_CART_STATUS -eq 0 ] && [ $TRAIN_CLUSTER_STATUS -eq 0 ]; then
    echo -e "${GREEN}All required files found!${NC}"
    echo ""
    if ! ask_yes_no "Regenerate any files?" "n"; then
        echo ""
        echo -e "${GREEN}Setup complete! You can now run sweeps.${NC}"
        echo ""
        echo "Next steps:"
        echo "  python sweep_launcher.py --sweep baseline --count 5"
        echo ""
        exit 0
    fi
    echo ""
fi

# =============================================================================
# Step 1: Generate Cartography Metrics for Training Set
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Step 1/4: Training Set Cartography Metrics"
echo -e "========================================================================${NC}"
echo ""
echo -e "${CYAN}Purpose:${NC} Track training dynamics to identify easy/hard/ambiguous examples"
echo -e "${CYAN}Output:${NC} ./cartography_output/cartography_metrics.csv"
echo -e "${CYAN}Time:${NC} ~10-15 minutes (${NUM_EPOCHS} epochs of training)"
echo ""

SKIP_TRAIN_CART=false
if [ -f "./cartography_output/cartography_metrics.csv" ]; then
    echo -e "${GREEN}✓ File already exists${NC}"
    if ! ask_yes_no "Regenerate training cartography metrics?" "n"; then
        SKIP_TRAIN_CART=true
    fi
else
    if ! ask_yes_no "Generate training cartography metrics?" "y"; then
        SKIP_TRAIN_CART=true
        echo -e "${YELLOW}⚠ Skipping. Move your file to: ./cartography_output/cartography_metrics.csv${NC}"
    fi
fi

if [ "$SKIP_TRAIN_CART" = false ]; then
    echo ""
    echo -e "${CYAN}Running cartography generation...${NC}"
if [ "$SKIP_TRAIN_CART" = false ]; then
    echo ""
    echo -e "${CYAN}Running cartography generation...${NC}"
    python3 run.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --max_length "${MAX_LENGTH}" \
      --do_train \
      --num_train_epochs "${NUM_EPOCHS}" \
      --per_device_train_batch_size ${PER_DEVICE_BATCHES} \
      --enable_cartography \
      --cartography_output_dir ./cartography_output \
      --output_dir ./temp_baseline_train \
      --eval_strategy epoch \
      --save_strategy epoch \
      --save_total_limit 1 \
      --logging_steps 100
    
    echo -e "${GREEN}✓ Training cartography metrics generated!${NC}"
fi

echo ""

# =============================================================================
# Step 2: Generate Cartography Metrics for Validation Set
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Step 2/4: Validation Set Cartography Metrics"
echo -e "========================================================================${NC}"
echo ""
echo -e "${CYAN}Purpose:${NC} Track validation dynamics for filtering validation set"
echo -e "${CYAN}Output:${NC} ./cartography_output_validation/cartography_metrics.csv"
echo -e "${CYAN}Time:${NC} ~5-10 minutes"
echo -e "${YELLOW}Note:${NC} Optional - only needed if you plan to filter validation set"
echo ""

SKIP_VAL_CART=false
if [ -f "./cartography_output_validation/cartography_metrics.csv" ]; then
    echo -e "${GREEN}✓ File already exists${NC}"
    if ! ask_yes_no "Regenerate validation cartography metrics?" "n"; then
        SKIP_VAL_CART=true
    fi
else
    if ! ask_yes_no "Generate validation cartography metrics?" "y"; then
        SKIP_VAL_CART=true
        echo -e "${YELLOW}⚠ Skipping. Move your file to: ./cartography_output_validation/cartography_metrics.csv${NC}"
    fi
fi

if [ "$SKIP_VAL_CART" = false ]; then
    echo ""
    echo -e "${CYAN}Running cartography generation...${NC}"
    python3 run.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --max_length "${MAX_LENGTH}" \
      --do_train \
      --train_split validation \
      --num_train_epochs "${NUM_EPOCHS}" \
      --per_device_train_batch_size ${PER_DEVICE_BATCHES} \
      --enable_cartography \
      --cartography_output_dir ./cartography_output_validation \
      --output_dir ./temp_baseline_val \
      --eval_strategy epoch \
      --save_strategy epoch \
      --save_total_limit 1 \
      --logging_steps 100
    
    echo -e "${GREEN}✓ Validation cartography metrics generated!${NC}"
fi

echo ""

# =============================================================================
# Step 3: Extract Embeddings for Clustering (Training Set)
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Step 3/4: Training Set Embeddings & Clusters"
echo -e "========================================================================${NC}"
echo ""
echo -e "${CYAN}Purpose:${NC} Create semantic clusters to identify similar examples"
echo -e "${CYAN}Output:${NC} ./cluster_output/cluster_assignments.csv"
echo -e "${CYAN}Time:${NC} ~3-5 minutes"
echo ""

SKIP_TRAIN_CLUSTER=false
if [ -f "./cluster_output/cluster_assignments.csv" ]; then
    echo -e "${GREEN}✓ File already exists${NC}"
    if ! ask_yes_no "Regenerate training cluster assignments?" "n"; then
        SKIP_TRAIN_CLUSTER=true
    fi
else
    if ! ask_yes_no "Generate training cluster assignments?" "y"; then
        SKIP_TRAIN_CLUSTER=true
        echo -e "${YELLOW}⚠ Skipping. Move your file to: ./cluster_output/cluster_assignments.csv${NC}"
    fi
fi

if [ "$SKIP_TRAIN_CLUSTER" = false ]; then
    echo ""
    echo -e "${CYAN}Step 3a: Extracting embeddings...${NC}"
if [ "$SKIP_TRAIN_CLUSTER" = false ]; then
    echo ""
    echo -e "${CYAN}Step 3a: Extracting embeddings...${NC}"
    
    # Use the trained model from step 1
    if [ -d "./temp_baseline_train" ]; then
        MODEL_PATH="./temp_baseline_train"
    else
        MODEL_PATH="${MODEL}"
        echo -e "${YELLOW}Warning: Using pre-trained model instead of fine-tuned model${NC}"
    fi
    
    python3 extract_embeddings.py \
      --model_path "${MODEL_PATH}" \
      --dataset "${DATASET}" \
      --split train \
      --output_dir ./embeddings_output \
      --max_samples 10000 \
      --batch_size 32 \
      --max_length "${MAX_LENGTH}"
    
    # Move/rename the output
    if [ -f "./embeddings_output/embeddings.npy" ]; then
        mv ./embeddings_output/embeddings.npy ./embeddings_train.npy
        echo -e "${GREEN}✓ Training embeddings extracted!${NC}"
    else
        echo -e "${RED}Error: Embeddings file not found!${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${CYAN}Step 3b: Generating clusters...${NC}"
    
    python3 cluster_analysis.py \
      --embeddings_path ./embeddings_train.npy \
      --output_dir ./cluster_output \
      --min_cluster_size 50 \
      --min_samples 10
    
    echo -e "${GREEN}✓ Cluster assignments generated!${NC}"
fi

echo ""

# =============================================================================
# Step 4: Generate Validation Set Clusters
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Step 4/4: Validation Set Embeddings & Clusters"
echo -e "========================================================================${NC}"
echo ""
echo -e "${CYAN}Purpose:${NC} Create semantic clusters for validation set filtering"
echo -e "${CYAN}Output:${NC} ./cluster_output_validation/cluster_assignments.csv"
echo -e "${CYAN}Time:${NC} ~2-3 minutes"
echo -e "${YELLOW}Note:${NC} Optional - only needed if you plan to filter validation set"
echo ""

SKIP_VAL_CLUSTER=false
if [ -f "./cluster_output_validation/cluster_assignments.csv" ]; then
    echo -e "${GREEN}✓ File already exists${NC}"
    if ! ask_yes_no "Regenerate validation cluster assignments?" "n"; then
        SKIP_VAL_CLUSTER=true
    fi
else
    if ! ask_yes_no "Generate validation cluster assignments?" "y"; then
        SKIP_VAL_CLUSTER=true
        echo -e "${YELLOW}⚠ Skipping. Move your file to: ./cluster_output_validation/cluster_assignments.csv${NC}"
    fi
fi

if [ "$SKIP_VAL_CLUSTER" = false ]; then
    echo ""
    echo -e "${CYAN}Step 4a: Extracting embeddings...${NC}"
    
    if [ -d "./temp_baseline_val" ]; then
        MODEL_PATH="./temp_baseline_val"
    else
        MODEL_PATH="${MODEL}"
    fi
    
    python3 extract_embeddings.py \
      --model_path "${MODEL_PATH}" \
      --dataset "${DATASET}" \
      --split validation \
      --output_dir ./embeddings_output_val \
      --batch_size 32 \
      --max_length "${MAX_LENGTH}"
    
    if [ -f "./embeddings_output_val/embeddings.npy" ]; then
        mv ./embeddings_output_val/embeddings.npy ./embeddings_validation.npy
        echo -e "${GREEN}✓ Validation embeddings extracted!${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Step 4b: Generating clusters...${NC}"
    
    python3 cluster_analysis.py \
      --embeddings_path ./embeddings_validation.npy \
      --output_dir ./cluster_output_validation \
      --min_cluster_size 20 \
      --min_samples 5
    
    echo -e "${GREEN}✓ Validation cluster assignments generated!${NC}"
fi

echo ""

# =============================================================================
# Cleanup
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Cleanup"
echo -e "========================================================================${NC}"
echo ""
echo "The following temporary directories were created:"
echo "  • ./temp_baseline_train"
echo "  • ./temp_baseline_val"
echo "  • ./embeddings_output"
echo "  • ./embeddings_output_val"
echo ""

if ask_yes_no "Remove temporary directories?" "y"; then
    rm -rf ./temp_baseline_train ./temp_baseline_val ./embeddings_output ./embeddings_output_val
    echo -e "${GREEN}✓ Cleanup complete!${NC}"
else
    echo -e "${YELLOW}Temporary directories kept.${NC}"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo -e "${BLUE}========================================================================"
echo "Setup Complete! 🎉"
echo -e "========================================================================${NC}"
echo ""
echo -e "${GREEN}Generated files:${NC}"

# Show what was actually generated
if [ -f "./cartography_output/cartography_metrics.csv" ]; then
    echo -e "  ${GREEN}✓${NC} ./cartography_output/cartography_metrics.csv"
else
    echo -e "  ${RED}✗${NC} ./cartography_output/cartography_metrics.csv ${YELLOW}(required)${NC}"
fi

if [ -f "./cluster_output/cluster_assignments.csv" ]; then
    echo -e "  ${GREEN}✓${NC} ./cluster_output/cluster_assignments.csv"
else
    echo -e "  ${RED}✗${NC} ./cluster_output/cluster_assignments.csv ${YELLOW}(required)${NC}"
fi

if [ -f "./cartography_output_validation/cartography_metrics.csv" ]; then
    echo -e "  ${GREEN}✓${NC} ./cartography_output_validation/cartography_metrics.csv"
else
    echo -e "  ${YELLOW}○${NC} ./cartography_output_validation/cartography_metrics.csv (optional)"
fi

if [ -f "./cluster_output_validation/cluster_assignments.csv" ]; then
    echo -e "  ${GREEN}✓${NC} ./cluster_output_validation/cluster_assignments.csv"
else
    echo -e "  ${YELLOW}○${NC} ./cluster_output_validation/cluster_assignments.csv (optional)"
fi

echo ""

# Check if we have minimum required files
if [ ! -f "./cartography_output/cartography_metrics.csv" ] || [ ! -f "./cluster_output/cluster_assignments.csv" ]; then
    echo -e "${RED}⚠ Warning: Missing required files!${NC}"
    echo ""
    echo "Please ensure the following files exist before running sweeps:"
    echo "  • ./cartography_output/cartography_metrics.csv"
    echo "  • ./cluster_output/cluster_assignments.csv"
    echo ""
    echo "You can either:"
    echo "  1. Run this script again and approve the required steps"
    echo "  2. Move your existing files to the paths shown above"
    echo ""
    exit 1
fi

echo -e "${CYAN}Next Steps:${NC}"
echo "------------------------------------------------------------------------"
echo ""
echo -e "${YELLOW}1. Run baseline experiments${NC} (recommended first):"
echo "   python sweep_launcher.py --sweep baseline --count 5"
echo ""
echo -e "${YELLOW}2. Run experimental sweeps:${NC}"
echo "   python sweep_launcher.py --sweep filtering --count 20"
echo "   python sweep_launcher.py --sweep smoothing --count 30"
echo "   python sweep_launcher.py --sweep combined --count 50"
echo ""
echo -e "${YELLOW}3. Analyze results:${NC}"
echo "   python analyze_sweep_results.py --compare_with_baseline"
echo ""
echo -e "${YELLOW}4. View W&B dashboard:${NC}"
echo "   https://wandb.ai"
echo ""
echo -e "${BLUE}========================================================================"
echo "Documentation"
echo -e "========================================================================${NC}"
echo ""
echo "• Quick Reference:     QUICKSTART.md"
echo "• Full Guide:          SWEEP_README.md"
echo "• Examples:            EXPERIMENT_EXAMPLES.sh"
echo "• Implementation Info: IMPLEMENTATION_SUMMARY.md"
echo ""
echo -e "${BLUE}========================================================================${NC}"

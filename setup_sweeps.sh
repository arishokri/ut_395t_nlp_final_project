#!/bin/bash
# Complete setup script for hyperparameter search system
# This script handles both initial setup and data preparation

set -e

DATASET="Eladio/emrqa-msquad"
MODEL="google/electra-small-discriminator"
NUM_EPOCHS=5
MAX_LENGTH=512

# =============================================================================
# Initial Setup: Dependencies and W&B Login
# =============================================================================

echo "========================================================================"
echo "Hyperparameter Search System - Setup"
echo "========================================================================"
echo ""

# Check if wandb is installed
echo "Checking dependencies..."
if ! python3 -c "import wandb" 2>/dev/null; then
    echo "Installing dependencies..."
    uv sync
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

echo ""

# Check if logged into wandb
echo "Checking W&B authentication..."
if ! wandb login --check 2>/dev/null; then
    echo "Please login to Weights & Biases:"
    wandb login
    echo "✓ Logged in to W&B"
else
    echo "✓ Already logged in to W&B"
fi

echo ""
echo "========================================================================"
echo "Data Preparation"
echo "========================================================================"
echo ""
echo "This will generate cartography metrics and cluster assignments."
echo "This is required for filtering, smoothing, and weighting experiments."
echo ""
read -p "Proceed with data preparation? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Setup complete! You can prepare data later by running:"
    echo "  ./setup_sweeps.sh"
    echo ""
    exit 0
fi

echo ""

# =============================================================================
# Step 1: Generate Cartography Metrics for Training Set
# =============================================================================

echo "Step 1/4: Generating cartography metrics for TRAINING set..."
echo "------------------------------------------------------------------------"

if [ -f "./cartography_output/cartography_metrics.csv" ]; then
    echo "Training cartography metrics already exist. Skipping..."
else
    python3 run.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --max_length "${MAX_LENGTH}" \
      --do_train \
      --num_train_epochs "${NUM_EPOCHS}" \
      --per_device_train_batch_size 16 \
      --enable_cartography \
      --cartography_output_dir ./cartography_output \
      --output_dir ./temp_baseline_train \
      --save_strategy epoch \
      --save_total_limit 1 \
      --logging_steps 100
    
    echo "✓ Training cartography metrics generated!"
fi

echo ""

# =============================================================================
# Step 2: Generate Cartography Metrics for Validation Set
# =============================================================================

echo "Step 2/4: Generating cartography metrics for VALIDATION set..."
echo "------------------------------------------------------------------------"

if [ -f "./cartography_output_validation/cartography_metrics.csv" ]; then
    echo "Validation cartography metrics already exist. Skipping..."
else
    python3 run.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --max_length "${MAX_LENGTH}" \
      --do_train \
      --train_split validation \
      --num_train_epochs "${NUM_EPOCHS}" \
      --per_device_train_batch_size 16 \
      --enable_cartography \
      --cartography_output_dir ./cartography_output_validation \
      --output_dir ./temp_baseline_val \
      --save_strategy epoch \
      --save_total_limit 1 \
      --logging_steps 100
    
    echo "✓ Validation cartography metrics generated!"
fi

echo ""

# =============================================================================
# Step 3: Extract Embeddings for Clustering (Training Set)
# =============================================================================

echo "Step 3/4: Extracting embeddings for TRAINING set..."
echo "------------------------------------------------------------------------"

if [ -f "./embeddings_train.npz" ]; then
    echo "Training embeddings already exist. Skipping..."
else
    # Use the trained model from step 1
    if [ -d "./temp_baseline_train" ]; then
        MODEL_PATH="./temp_baseline_train"
    else
        MODEL_PATH="${MODEL}"
        echo "Warning: Using pre-trained model instead of fine-tuned model"
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
    if [ -f "./embeddings_output/embeddings.npz" ]; then
        mv ./embeddings_output/embeddings.npz ./embeddings_train.npz
        echo "✓ Training embeddings extracted!"
    else
        echo "Error: Embeddings file not found!"
        exit 1
    fi
fi

echo ""

# =============================================================================
# Step 4: Generate Cluster Assignments (Training Set)
# =============================================================================

echo "Step 4/4: Generating cluster assignments for TRAINING set..."
echo "------------------------------------------------------------------------"

if [ -f "./cluster_output/cluster_assignments.csv" ]; then
    echo "Cluster assignments already exist. Skipping..."
else
    python3 cluster_analysis.py \
      --embeddings_path ./embeddings_train.npz \
      --output_dir ./cluster_output \
      --min_cluster_size 50 \
      --min_samples 10
    
    echo "✓ Cluster assignments generated!"
fi

echo ""

# =============================================================================
# Optional: Generate Validation Set Clusters
# =============================================================================

echo "Optional: Generating cluster assignments for VALIDATION set..."
echo "------------------------------------------------------------------------"
read -p "Generate validation cluster assignments? (recommended) (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    
    # Extract validation embeddings
    if [ ! -f "./embeddings_validation.npz" ]; then
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
        
        if [ -f "./embeddings_output_val/embeddings.npz" ]; then
            mv ./embeddings_output_val/embeddings.npz ./embeddings_validation.npz
        fi
    fi
    
    # Generate clusters
    if [ ! -f "./cluster_output_validation/cluster_assignments.csv" ]; then
        python3 cluster_analysis.py \
          --embeddings_path ./embeddings_validation.npz \
          --output_dir ./cluster_output_validation \
          --min_cluster_size 20 \
          --min_samples 5
    fi
    
    echo "✓ Validation cluster assignments generated!"
else
    echo "Skipping validation cluster generation."
fi

echo ""

# =============================================================================
# Cleanup
# =============================================================================

read -p "Cleanup temporary directories? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ./temp_baseline_train ./temp_baseline_val ./embeddings_output ./embeddings_output_val
    echo "✓ Cleanup complete!"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "========================================================================"
echo "Setup Complete! 🎉"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  ✓ ./cartography_output/cartography_metrics.csv"
echo "  ✓ ./cartography_output_validation/cartography_metrics.csv"
echo "  ✓ ./embeddings_train.npz"
echo "  ✓ ./cluster_output/cluster_assignments.csv"

if [ -f "./cluster_output_validation/cluster_assignments.csv" ]; then
    echo "  ✓ ./embeddings_validation.npz"
    echo "  ✓ ./cluster_output_validation/cluster_assignments.csv"
fi

echo ""
echo "Next Steps:"
echo "------------------------------------------------------------------------"
echo ""
echo "1. Run baseline experiments (recommended first):"
echo "   python sweep_launcher.py --sweep baseline --count 5"
echo ""
echo "2. Run experimental sweeps:"
echo "   python sweep_launcher.py --sweep filtering --count 20"
echo "   python sweep_launcher.py --sweep smoothing --count 30"
echo "   python sweep_launcher.py --sweep combined --count 50"
echo ""
echo "3. Analyze results:"
echo "   python analyze_sweep_results.py --compare_with_baseline"
echo ""
echo "4. View W&B dashboard:"
echo "   https://wandb.ai"
echo ""
echo "========================================================================"
echo "Documentation"
echo "========================================================================"
echo ""
echo "• Quick Reference:     QUICKSTART.md"
echo "• Full Guide:          SWEEP_README.md"
echo "• Examples:            EXPERIMENT_EXAMPLES.sh"
echo "• Implementation Info: IMPLEMENTATION_SUMMARY.md"
echo ""
echo "========================================================================"

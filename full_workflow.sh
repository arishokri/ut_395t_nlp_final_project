#!/bin/bash
# Complete workflow example for clustering analysis
# This script demonstrates the full pipeline from training to analysis

set -e  # Exit on error

echo "=========================================================================="
echo "COMPLETE CLUSTERING ANALYSIS WORKFLOW"
echo "=========================================================================="

# Configuration
MODEL_DIR="./trained_model_example"
CART_DIR="./cartography_example"
EMB_DIR="./embeddings_example"
CLUSTER_DIR="./cluster_example"
ANALYSIS_DIR="./analysis_example"
MAX_SAMPLES=5000  # Use subset for faster testing

echo ""
echo "Configuration:"
echo "  Model output: $MODEL_DIR"
echo "  Cartography: $CART_DIR"
echo "  Embeddings: $EMB_DIR"
echo "  Clusters: $CLUSTER_DIR"
echo "  Analysis: $ANALYSIS_DIR"
echo "  Sample size: $MAX_SAMPLES"
echo ""

# =============================================================================
# STEP 1: Train model with cartography enabled
# =============================================================================
echo "=========================================================================="
echo "STEP 1: Training model with cartography enabled"
echo "=========================================================================="

python run.py \
  --do_train \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --output_dir "$MODEL_DIR" \
  --enable_cartography \
  --cartography_output_dir "$CART_DIR" \
  --max_train_samples "$MAX_SAMPLES" \
  --save_strategy epoch \
  --logging_steps 100

echo "✓ Training complete with cartography metrics saved"
echo ""

# =============================================================================
# STEP 2: Extract embeddings from trained model
# =============================================================================
echo "=========================================================================="
echo "STEP 2: Extracting [CLS + answer span] embeddings from trained model"
echo "=========================================================================="

python extract_embeddings.py \
  --model_path "$MODEL_DIR" \
  --dataset Eladio/emrqa-msquad \
  --output_dir "$EMB_DIR" \
  --split train \
  --max_samples "$MAX_SAMPLES" \
  --batch_size 32

echo "✓ Embeddings extracted successfully"
echo ""

# =============================================================================
# STEP 3: Perform clustering analysis with HDBSCAN
# =============================================================================
echo "=========================================================================="
echo "STEP 3: HDBSCAN Clustering (PCA → HDBSCAN → UMAP visualization)"
echo "=========================================================================="

# HDBSCAN with 50D PCA reduction
echo "3a. HDBSCAN clustering with 50D PCA reduction..."
python cluster_analysis.py \
  --embedding_dir "$EMB_DIR" \
  --output_dir "${CLUSTER_DIR}_50d" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --reduction_dim 50 \
  --reduction_method pca \
  --min_cluster_size 15 \
  --visualization_method umap

echo "✓ 50D clustering complete"
echo ""

# Try different reduction dimensions
echo "3b. HDBSCAN with 30D PCA (more conservative)..."
python cluster_analysis.py \
  --embedding_dir "$EMB_DIR" \
  --output_dir "${CLUSTER_DIR}_30d" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --reduction_dim 30 \
  --reduction_method pca \
  --min_cluster_size 15 \
  --visualization_method umap

echo "✓ 30D clustering complete"
echo ""

# =============================================================================
# STEP 4: Integrated analysis
# =============================================================================
echo "=========================================================================="
echo "STEP 4: Integrated cartography + clustering analysis"
echo "=========================================================================="

# Analyze 50D results with cartography
echo "4a. Analyzing 50D clustering + cartography..."
python analyze_dataset.py \
  --cartography_dir "$CART_DIR" \
  --cluster_dir "${CLUSTER_DIR}_50d" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --output_dir "${ANALYSIS_DIR}_50d"

echo "✓ 50D analysis complete"
echo ""

# Analyze 30D results with cartography
echo "4b. Analyzing 30D clustering + cartography..."
python analyze_dataset.py \
  --cartography_dir "$CART_DIR" \
  --cluster_dir "${CLUSTER_DIR}_30d" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --output_dir "${ANALYSIS_DIR}_30d"

echo "✓ 30D analysis complete"
echo ""

# =============================================================================
# STEP 5: Summary of results
# =============================================================================
echo "=========================================================================="
echo "WORKFLOW COMPLETE - SUMMARY OF OUTPUTS"
echo "=========================================================================="

echo ""
echo "Model & Training:"
echo "  ├─ Model checkpoint: $MODEL_DIR/"
echo "  └─ Cartography metrics: $CART_DIR/"
echo ""

echo "Embeddings:"
echo "  └─ [CLS + answer span]: ${EMB_DIR}/embeddings.npy"
echo ""

echo "Clustering Results:"
echo "  ├─ 50D: ${CLUSTER_DIR}_50d/"
echo "  └─ 30D: ${CLUSTER_DIR}_30d/"
echo ""

echo "Integrated Analysis:"
echo "  ├─ 50D analysis: ${ANALYSIS_DIR}_50d/"
echo "  └─ 30D analysis: ${ANALYSIS_DIR}_30d/"
echo ""

echo "=========================================================================="
echo "KEY FILES TO EXAMINE:"
echo "=========================================================================="
echo ""
echo "1. START HERE - Most problematic clusters:"
echo "   ${ANALYSIS_DIR}_50d/interesting_patterns.json"
echo ""
echo "2. Category-cluster overlap:"
echo "   ${ANALYSIS_DIR}_50d/category_cluster_overlap.csv"
echo ""
echo "3. Cluster statistics (with HDBSCAN persistence):"
echo "   ${CLUSTER_DIR}_50d/cluster_statistics.csv"
echo ""
echo "4. Example questions per cluster:"
echo "   ${CLUSTER_DIR}_50d/cluster_samples.json"
echo ""
echo "5. Visualizations:"
echo "   ${ANALYSIS_DIR}_50d/integrated_scatter.png"
echo "   ${ANALYSIS_DIR}_50d/category_cluster_heatmap.png"
echo "   ${CLUSTER_DIR}_50d/cluster_visualization.png"
echo "   ${CLUSTER_DIR}_50d/cluster_persistence.png (HDBSCAN stability)"
echo "   ${CLUSTER_DIR}_50d/cluster_probabilities.png"
echo ""

echo "=========================================================================="
echo "NEXT STEPS:"
echo "=========================================================================="
echo ""
echo "1. Review interesting_patterns.json to identify problematic clusters"
echo "2. Examine cluster_samples.json to understand what makes each cluster unique"
echo "3. Check cluster persistence scores (higher = more stable clusters)"
echo "4. Examine cluster probabilities to find uncertain examples"
echo "5. Compare 50D vs 30D results to find optimal dimensionality"
echo "6. Compare full embeddings vs question-only embeddings"
echo "7. Use findings to:"
echo "   - Identify data augmentation needs"
echo "   - Find annotation inconsistencies"
echo "   - Understand model's failure modes"
echo "   - Guide targeted improvements"
echo "   - Focus on low-probability cluster members (potential outliers)"
echo ""
echo "=========================================================================="

# Optional: Print quick stats
echo "Quick Statistics:"
echo "=========================================================================="
echo ""

if [ -f "${ANALYSIS_DIR}_50d/interesting_patterns.json" ]; then
    echo "Most problematic clusters (50D):"
    cat "${ANALYSIS_DIR}_50d/interesting_patterns.json" | python -m json.tool | grep -A 5 "most_problematic_clusters"
    echo ""
fi

if [ -f "${CLUSTER_DIR}_50d/cluster_statistics.csv" ]; then
    echo "Cluster sizes and persistence (50D):"
    head -n 11 "${CLUSTER_DIR}_50d/cluster_statistics.csv" | column -t -s,
    echo ""
fi

echo "=========================================================================="
echo "Workflow completed successfully!"
echo "=========================================================================="

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
echo "STEP 2: Extracting embeddings from trained model"
echo "=========================================================================="

# Extract both CLS and question-only embeddings for comparison
python extract_embeddings.py \
  --model_path "$MODEL_DIR" \
  --dataset Eladio/emrqa-msquad \
  --output_dir "$EMB_DIR" \
  --split train \
  --max_samples "$MAX_SAMPLES" \
  --batch_size 32 \
  --embedding_types cls question_only

echo "✓ Embeddings extracted successfully"
echo ""

# =============================================================================
# STEP 3: Perform clustering analysis
# =============================================================================
echo "=========================================================================="
echo "STEP 3: Clustering embeddings"
echo "=========================================================================="

# Try K-means with automatic optimal k detection
echo "3a. K-means clustering with optimal k detection..."
python cluster_analysis.py \
  --embedding_dir "$EMB_DIR" \
  --output_dir "${CLUSTER_DIR}_kmeans" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --embedding_type cls \
  --clustering_method kmeans \
  --find_optimal \
  --reduction_method tsne

echo "✓ K-means clustering complete"
echo ""

# Also try DBSCAN to find outliers
echo "3b. DBSCAN clustering for outlier detection..."
python cluster_analysis.py \
  --embedding_dir "$EMB_DIR" \
  --output_dir "${CLUSTER_DIR}_dbscan" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --embedding_type cls \
  --clustering_method dbscan \
  --dbscan_eps 0.4 \
  --dbscan_min_samples 15 \
  --reduction_method umap

echo "✓ DBSCAN clustering complete"
echo ""

# Try question-only embeddings
echo "3c. Question-only embeddings clustering..."
python cluster_analysis.py \
  --embedding_dir "$EMB_DIR" \
  --output_dir "${CLUSTER_DIR}_question_only" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --embedding_type question_only \
  --clustering_method kmeans \
  --n_clusters 10 \
  --reduction_method umap

echo "✓ Question-only clustering complete"
echo ""

# =============================================================================
# STEP 4: Integrated analysis
# =============================================================================
echo "=========================================================================="
echo "STEP 4: Integrated cartography + clustering analysis"
echo "=========================================================================="

# Analyze K-means results with cartography
echo "4a. Analyzing K-means + cartography..."
python analyze_dataset.py \
  --cartography_dir "$CART_DIR" \
  --cluster_dir "${CLUSTER_DIR}_kmeans" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --output_dir "${ANALYSIS_DIR}_kmeans"

echo "✓ K-means analysis complete"
echo ""

# Analyze DBSCAN results with cartography
echo "4b. Analyzing DBSCAN + cartography..."
python analyze_dataset.py \
  --cartography_dir "$CART_DIR" \
  --cluster_dir "${CLUSTER_DIR}_dbscan" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --output_dir "${ANALYSIS_DIR}_dbscan"

echo "✓ DBSCAN analysis complete"
echo ""

# Analyze question-only clustering
echo "4c. Analyzing question-only clustering + cartography..."
python analyze_dataset.py \
  --cartography_dir "$CART_DIR" \
  --cluster_dir "${CLUSTER_DIR}_question_only" \
  --dataset Eladio/emrqa-msquad \
  --split train \
  --output_dir "${ANALYSIS_DIR}_question_only"

echo "✓ Question-only analysis complete"
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
echo "  ├─ CLS embeddings: ${EMB_DIR}/embeddings_cls.npy"
echo "  └─ Question-only: ${EMB_DIR}/embeddings_question_only.npy"
echo ""

echo "Clustering Results:"
echo "  ├─ K-means: ${CLUSTER_DIR}_kmeans/"
echo "  ├─ DBSCAN: ${CLUSTER_DIR}_dbscan/"
echo "  └─ Question-only: ${CLUSTER_DIR}_question_only/"
echo ""

echo "Integrated Analysis:"
echo "  ├─ K-means analysis: ${ANALYSIS_DIR}_kmeans/"
echo "  ├─ DBSCAN analysis: ${ANALYSIS_DIR}_dbscan/"
echo "  └─ Question-only: ${ANALYSIS_DIR}_question_only/"
echo ""

echo "=========================================================================="
echo "KEY FILES TO EXAMINE:"
echo "=========================================================================="
echo ""
echo "1. START HERE - Most problematic clusters:"
echo "   ${ANALYSIS_DIR}_kmeans/interesting_patterns.json"
echo ""
echo "2. Category-cluster overlap:"
echo "   ${ANALYSIS_DIR}_kmeans/category_cluster_overlap.csv"
echo ""
echo "3. Cluster statistics:"
echo "   ${CLUSTER_DIR}_kmeans/cluster_statistics.csv"
echo ""
echo "4. Example questions per cluster:"
echo "   ${CLUSTER_DIR}_kmeans/cluster_samples.json"
echo ""
echo "5. Visualizations:"
echo "   ${ANALYSIS_DIR}_kmeans/integrated_scatter.png"
echo "   ${ANALYSIS_DIR}_kmeans/category_cluster_heatmap.png"
echo "   ${CLUSTER_DIR}_kmeans/cluster_visualization.png"
echo ""

echo "=========================================================================="
echo "NEXT STEPS:"
echo "=========================================================================="
echo ""
echo "1. Review interesting_patterns.json to identify problematic clusters"
echo "2. Examine cluster_samples.json to understand what makes each cluster unique"
echo "3. Compare K-means vs DBSCAN results"
echo "4. Compare full embeddings vs question-only embeddings"
echo "5. Use findings to:"
echo "   - Identify data augmentation needs"
echo "   - Find annotation inconsistencies"
echo "   - Understand model's failure modes"
echo "   - Guide targeted improvements"
echo ""
echo "=========================================================================="

# Optional: Print quick stats
echo "Quick Statistics:"
echo "=========================================================================="
echo ""

if [ -f "${ANALYSIS_DIR}_kmeans/interesting_patterns.json" ]; then
    echo "Most problematic clusters (K-means):"
    cat "${ANALYSIS_DIR}_kmeans/interesting_patterns.json" | python -m json.tool | grep -A 5 "most_problematic_clusters"
    echo ""
fi

if [ -f "${CLUSTER_DIR}_kmeans/cluster_statistics.csv" ]; then
    echo "Cluster sizes (K-means):"
    head -n 11 "${CLUSTER_DIR}_kmeans/cluster_statistics.csv" | column -t -s,
    echo ""
fi

echo "=========================================================================="
echo "Workflow completed successfully!"
echo "=========================================================================="

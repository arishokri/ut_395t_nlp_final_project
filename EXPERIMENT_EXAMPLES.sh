# Example experiment configurations
# Copy and modify these commands for your experiments

# =============================================================================
# BASELINE EXPERIMENTS
# =============================================================================

# Standard baseline
./run_experiment.sh "baseline_seed42" --seed 42

# Multiple seeds for statistical significance
./run_experiment.sh "baseline_seed123" --seed 123
./run_experiment.sh "baseline_seed456" --seed 456

# =============================================================================
# CARTOGRAPHY FILTERING EXPERIMENTS
# =============================================================================

# Enable cartography tracking (required for filtering)
./run_experiment.sh "cartography_only" \
  --enable_cartography \
  --cartography_output_dir ./cartography_output

# Filter out ambiguous examples
./run_experiment.sh "filtered_ambiguous" \
  --enable_cartography \
  --filter_cartography \
  --cartography_output_dir ./cartography_output

# Filter both training and validation
./run_experiment.sh "filtered_train_val" \
  --enable_cartography \
  --filter_cartography \
  --filter_validation \
  --cartography_output_dir ./cartography_output \
  --validation_cartography_output_dir ./cartography_output_validation

# =============================================================================
# CLUSTER FILTERING EXPERIMENTS
# =============================================================================

# Exclude noise cluster (-1)
./run_experiment.sh "cluster_no_noise" \
  --filter_clusters \
  --exclude_clusters "-1" \
  --cluster_assignments_path ./cluster_output

# Exclude multiple clusters
./run_experiment.sh "cluster_exclude_3_4" \
  --filter_clusters \
  --exclude_clusters "3,4,-1" \
  --cluster_assignments_path ./cluster_output

# Filter by cluster probability
./run_experiment.sh "cluster_prob_0.7" \
  --filter_clusters \
  --exclude_clusters "-1" \
  --min_cluster_probability 0.7 \
  --cluster_assignments_path ./cluster_output

# =============================================================================
# LABEL SMOOTHING EXPERIMENTS
# =============================================================================

# Label smoothing (requires pre-computed cartography metrics)
./run_experiment.sh "smooth_0.6" \
  --use_label_smoothing \
  --smoothing_factor 0.6 \
  --cartography_output_dir ./cartography_output

# Different smoothing factors
./run_experiment.sh "smooth_0.4" \
  --use_label_smoothing \
  --smoothing_factor 0.4 \
  --cartography_output_dir ./cartography_output

./run_experiment.sh "smooth_0.8" \
  --use_label_smoothing \
  --smoothing_factor 0.8 \
  --cartography_output_dir ./cartography_output

# =============================================================================
# SOFT WEIGHTING EXPERIMENTS
# =============================================================================

# Soft weighting based on variability
./run_experiment.sh "soft_weight" \
  --use_soft_weighting \
  --weight_clip_min 0.1 \
  --weight_clip_max 10.0 \
  --cartography_output_dir ./cartography_output

# More aggressive weighting
./run_experiment.sh "soft_weight_aggressive" \
  --use_soft_weighting \
  --weight_clip_min 0.05 \
  --weight_clip_max 20.0 \
  --cartography_output_dir ./cartography_output

# =============================================================================
# COMBINED STRATEGIES
# =============================================================================

# Cartography filtering + Label smoothing
./run_experiment.sh "cart_filt_smooth" \
  --filter_cartography \
  --use_label_smoothing \
  --smoothing_factor 0.6 \
  --cartography_output_dir ./cartography_output

# Cluster filtering + Soft weighting
./run_experiment.sh "cluster_filt_weight" \
  --filter_clusters \
  --exclude_clusters "-1" \
  --use_soft_weighting \
  --weight_clip_min 0.1 \
  --weight_clip_max 10.0 \
  --cluster_assignments_path ./cluster_output \
  --cartography_output_dir ./cartography_output

# All strategies combined
./run_experiment.sh "all_strategies" \
  --filter_cartography \
  --filter_clusters \
  --exclude_clusters "-1" \
  --min_cluster_probability 0.5 \
  --use_label_smoothing \
  --smoothing_factor 0.6 \
  --use_soft_weighting \
  --weight_clip_min 0.1 \
  --weight_clip_max 10.0 \
  --filter_validation \
  --cartography_output_dir ./cartography_output \
  --validation_cartography_output_dir ./cartography_output_validation \
  --cluster_assignments_path ./cluster_output \
  --validation_cluster_assignments_path ./cluster_output_validation

# =============================================================================
# ABLATION STUDIES
# =============================================================================

# Question-only ablation
./run_experiment.sh "ablation_question_only" \
  --ablations q_only

# Passage-only ablation  
./run_experiment.sh "ablation_passage_only" \
  --ablations p_only

# =============================================================================
# WANDB TAGGING EXAMPLES
# =============================================================================

# Add tags to organize experiments
./run_experiment.sh "baseline_tagged" \
  --wandb_tags "baseline,initial-run,seed42" \
  --seed 42

./run_experiment.sh "best_config" \
  --filter_cartography \
  --use_label_smoothing \
  --smoothing_factor 0.6 \
  --wandb_tags "best-performer,production-candidate" \
  --cartography_output_dir ./cartography_output

# =============================================================================
# HYPERPARAMETER SWEEP USAGE
# =============================================================================

# Instead of running individual experiments, use sweeps:

# Launch baseline sweep
python sweep_launcher.py --sweep baseline --count 5

# Launch filtering strategies sweep
python sweep_launcher.py --sweep filtering --count 20

# Launch smoothing/weighting sweep
python sweep_launcher.py --sweep smoothing --count 30

# Launch combined strategies sweep with Bayesian optimization
python sweep_launcher.py --sweep combined --count 50

# Resume an existing sweep
python sweep_launcher.py --resume <sweep_id> --count 10

# =============================================================================
# ANALYSIS COMMANDS
# =============================================================================

# Analyze all experiments
python analyze_sweep_results.py --compare_with_baseline

# Export results to CSV
python analyze_sweep_results.py --export all_results.csv

# Analyze specific sweeps
python analyze_sweep_results.py --sweep_ids abc123 def456

# List all sweeps
python sweep_launcher.py --list

# Check sweep status
python sweep_launcher.py --status <sweep_id>

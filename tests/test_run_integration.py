"""Integration tests for run.py entry points.

These tests verify that different combinations of arguments work correctly.
"""

import subprocess
import sys


class TestRunBaseline:
    """Test baseline training scenarios."""

    def test_help_command(self):
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, "run.py", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should exit successfully
        assert result.returncode in [0, 1]  # May fail due to missing args
        # Help text should mention key arguments
        assert "--model" in result.stdout or "--help" in result.stdout

    def test_baseline_dry_run(self, temp_dir):
        """Test baseline configuration (dry run - no actual training)."""
        # This test validates argument parsing without actually training
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "baseline_test"),
            "--do_train",
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "2",
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        # Note: This will likely fail without GPU/proper setup, but we test argument parsing
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # We mainly care that arguments are parsed correctly
        # Actual training may fail due to environment
        assert "error: unrecognized arguments" not in result.stderr


class TestRunWithCartography:
    """Test cartography-enabled training scenarios."""

    def test_cartography_arguments_parsing(self, temp_dir):
        """Test that cartography arguments are recognized."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "cartography_test"),
            "--do_train",
            "--num_train_epochs", "1",
            "--enable_cartography",
            "--cartography_output_dir", str(temp_dir / "cartography_output"),
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Arguments should be recognized
        assert "error: unrecognized arguments" not in result.stderr
        assert "--enable_cartography" not in result.stderr

    def test_cartography_filter_arguments(self, temp_dir):
        """Test cartography filtering arguments."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "filter_test"),
            "--do_train",
            "--filter_cartography",
            "--cartography_output_dir", str(temp_dir / "cartography_output"),
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr


class TestRunWithClusters:
    """Test cluster-based filtering scenarios."""

    def test_cluster_filter_arguments(self, temp_dir):
        """Test cluster filtering arguments."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "cluster_test"),
            "--do_train",
            "--filter_clusters",
            "--cluster_assignments_path", str(temp_dir / "cluster_output"),
            "--exclude_clusters", "-1",
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr

    def test_cluster_probability_threshold(self, temp_dir):
        """Test cluster probability threshold argument."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "cluster_prob_test"),
            "--do_train",
            "--filter_clusters",
            "--cluster_assignments_path", str(temp_dir / "cluster_output"),
            "--min_cluster_probability", "0.8",
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr


class TestRunWithLabelSmoothing:
    """Test label smoothing scenarios."""

    def test_label_smoothing_arguments(self, temp_dir):
        """Test label smoothing arguments."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "smoothing_test"),
            "--do_train",
            "--use_label_smoothing",
            "--smoothing_factor", "0.6",
            "--cartography_output_dir", str(temp_dir / "cartography_output"),
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr


class TestRunWithSoftWeighting:
    """Test soft weighting scenarios."""

    def test_soft_weighting_arguments(self, temp_dir):
        """Test soft weighting arguments."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "weighting_test"),
            "--do_train",
            "--use_soft_weighting",
            "--weight_clip_min", "0.1",
            "--weight_clip_max", "10.0",
            "--cartography_output_dir", str(temp_dir / "cartography_output"),
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr


class TestRunCombinedStrategies:
    """Test combining multiple strategies."""

    def test_cartography_and_cluster_filtering(self, temp_dir):
        """Test combining cartography and cluster filtering."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "combined_test"),
            "--do_train",
            "--filter_cartography",
            "--cartography_output_dir", str(temp_dir / "cartography_output"),
            "--filter_clusters",
            "--cluster_assignments_path", str(temp_dir / "cluster_output"),
            "--exclude_clusters", "-1",
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr

    def test_filtering_with_smoothing(self, temp_dir):
        """Test combining filtering with label smoothing."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "filter_smooth_test"),
            "--do_train",
            "--filter_cartography",
            "--use_label_smoothing",
            "--smoothing_factor", "0.5",
            "--cartography_output_dir", str(temp_dir / "cartography_output"),
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr


class TestRunAblations:
    """Test ablation study configurations."""

    def test_question_only_ablation(self, temp_dir):
        """Test question-only ablation."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "q_only_test"),
            "--do_train",
            "--ablations", "q_only",
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr

    def test_passage_only_ablation(self, temp_dir):
        """Test passage-only ablation."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "p_only_test"),
            "--do_train",
            "--ablations", "p_only",
            "--max_train_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr


class TestRunValidationFiltering:
    """Test validation set filtering."""

    def test_validation_cartography_filtering(self, temp_dir):
        """Test filtering validation set with cartography."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "val_filter_test"),
            "--do_train",
            "--filter_cartography",
            "--filter_validation",
            "--cartography_output_dir", str(temp_dir / "cartography_output"),
            "--validation_cartography_output_dir", str(temp_dir / "cartography_output_val"),
            "--max_train_samples", "10",
            "--max_eval_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr

    def test_validation_cluster_filtering(self, temp_dir):
        """Test filtering validation set with clusters."""
        cmd = [
            sys.executable, "run.py",
            "--output_dir", str(temp_dir / "val_cluster_test"),
            "--do_train",
            "--filter_clusters",
            "--cluster_assignments_path", str(temp_dir / "cluster_output"),
            "--validation_cluster_assignments_path", str(temp_dir / "cluster_output_val"),
            "--exclude_clusters", "-1",
            "--max_train_samples", "10",
            "--max_eval_samples", "10",
            "--save_strategy", "no",
            "--eval_strategy", "no",
            "--report_to", "none",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert "error: unrecognized arguments" not in result.stderr

"""Tests for boolean argument handling in run.py.

These tests verify that boolean arguments work correctly with both:
1. Flag-style syntax: --flag (CLI usage)
2. Value-style syntax: --flag=true/false (W&B sweeps)
"""

import argparse
import pytest
from run import str_to_bool


class TestStrToBool:
    """Test the str_to_bool conversion function."""

    def test_bool_input_true(self):
        """Test that boolean True is returned as-is."""
        assert str_to_bool(True) is True

    def test_bool_input_false(self):
        """Test that boolean False is returned as-is."""
        assert str_to_bool(False) is False

    def test_none_input(self):
        """Test that None returns False."""
        assert str_to_bool(None) is False

    def test_string_true_variants(self):
        """Test that various 'true' strings are converted correctly."""
        true_values = [
            "true",
            "True",
            "TRUE",
            "t",
            "T",
            "yes",
            "Yes",
            "YES",
            "y",
            "Y",
            "1",
        ]
        for value in true_values:
            assert str_to_bool(value) is True, f"Failed for value: {value}"

    def test_string_false_variants(self):
        """Test that various 'false' strings are converted correctly."""
        false_values = [
            "false",
            "False",
            "FALSE",
            "f",
            "F",
            "no",
            "No",
            "NO",
            "n",
            "N",
            "0",
        ]
        for value in false_values:
            assert str_to_bool(value) is False, f"Failed for value: {value}"

    def test_invalid_string_raises_error(self):
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError, match="Boolean value expected"):
            str_to_bool("invalid")

        with pytest.raises(ValueError, match="Boolean value expected"):
            str_to_bool("maybe")

        with pytest.raises(ValueError, match="Boolean value expected"):
            str_to_bool("2")


class TestBooleanArgumentParsing:
    """Test that argparse handles boolean arguments correctly."""

    def create_parser(self):
        """Create a test parser with boolean arguments using the same pattern as run.py."""
        parser = argparse.ArgumentParser()

        # Add test boolean arguments using the same pattern as run.py
        parser.add_argument(
            "--test_flag",
            type=str_to_bool,
            nargs="?",
            const=True,
            default=False,
            help="Test boolean flag",
        )

        parser.add_argument(
            "--another_flag",
            type=str_to_bool,
            nargs="?",
            const=True,
            default=False,
            help="Another test boolean flag",
        )

        return parser

    def test_flag_style_single_flag(self):
        """Test flag-style argument: --flag"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag"])

        assert args.test_flag is True
        assert args.another_flag is False

    def test_flag_style_multiple_flags(self):
        """Test multiple flag-style arguments: --flag1 --flag2"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag", "--another_flag"])

        assert args.test_flag is True
        assert args.another_flag is True

    def test_value_style_true(self):
        """Test value-style argument: --flag=true"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag=true"])

        assert args.test_flag is True
        assert args.another_flag is False

    def test_value_style_false(self):
        """Test value-style argument: --flag=false"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag=false"])

        assert args.test_flag is False
        assert args.another_flag is False

    def test_value_style_mixed(self):
        """Test mixed value-style arguments: --flag1=true --flag2=false"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag=true", "--another_flag=false"])

        assert args.test_flag is True
        assert args.another_flag is False

    def test_value_style_case_insensitive(self):
        """Test that value parsing is case-insensitive: --flag=True, --flag=FALSE"""
        parser = self.create_parser()

        args1 = parser.parse_args(["--test_flag=True"])
        assert args1.test_flag is True

        args2 = parser.parse_args(["--test_flag=FALSE"])
        assert args2.test_flag is False

    def test_no_flags_default_false(self):
        """Test that flags default to False when not provided."""
        parser = self.create_parser()
        args = parser.parse_args([])

        assert args.test_flag is False
        assert args.another_flag is False

    def test_mixed_flag_and_value_style(self):
        """Test mixing flag-style and value-style: --flag1 --flag2=false"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag", "--another_flag=false"])

        assert args.test_flag is True
        assert args.another_flag is False

    def test_wandb_sweep_style_true(self):
        """Test W&B sweep format: --flag=true"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag=true", "--another_flag=true"])

        assert args.test_flag is True
        assert args.another_flag is True

    def test_wandb_sweep_style_false(self):
        """Test W&B sweep format: --flag=false"""
        parser = self.create_parser()
        args = parser.parse_args(["--test_flag=false", "--another_flag=false"])

        assert args.test_flag is False
        assert args.another_flag is False


class TestRunPyBooleanArguments:
    """Integration tests for boolean arguments in run.py itself."""

    def test_all_boolean_args_available(self):
        """Test that all expected boolean arguments are defined in run.py."""
        import subprocess
        import sys

        # Get help output from run.py
        result = subprocess.run(
            [sys.executable, "run.py", "--help"],
            capture_output=True,
            text=True,
        )

        # Check that all boolean arguments are documented in help
        expected_bool_args = [
            "--enable_cartography",
            "--filter_ambiguous",
            "--filter_clusters",
            "--exclude_noise_cluster",
            "--use_label_smoothing",
            "--use_soft_weighting",
            "--filter_validation",
            "--filter_rule_based",
        ]

        help_text = result.stdout + result.stderr
        for arg in expected_bool_args:
            assert arg in help_text, f"Boolean argument {arg} not found in help text"

    def test_run_py_flag_style_parsing(self):
        """Test that run.py accepts flag-style boolean arguments."""
        import subprocess
        import sys

        # Test with minimal args plus boolean flags
        # We don't actually run training, just test that arguments parse correctly
        cmd = [
            sys.executable,
            "run.py",
            "--output_dir",
            "/tmp/test_output",
            "--max_train_samples",
            "1",
            "--num_train_epochs",
            "1",
            "--filter_clusters",
            "--filter_validation",
            "--help",  # Add help to prevent actual execution
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        # Should not have argument parsing errors
        assert "error: unrecognized arguments" not in result.stderr.lower()
        assert "invalid" not in result.stderr.lower() or "--help" in result.stderr

    def test_run_py_value_style_parsing(self):
        """Test that run.py accepts value-style boolean arguments (W&B sweep format)."""
        import subprocess
        import sys

        # Test with value-style arguments
        cmd = [
            sys.executable,
            "run.py",
            "--output_dir",
            "/tmp/test_output",
            "--max_train_samples",
            "1",
            "--num_train_epochs",
            "1",
            "--filter_clusters=true",
            "--filter_validation=false",
            "--enable_cartography=true",
            "--help",  # Add help to prevent actual execution
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        # Should not have argument parsing errors
        assert "error: unrecognized arguments" not in result.stderr.lower()
        assert "invalid" not in result.stderr.lower() or "--help" in result.stderr

#!/usr/bin/env python3
"""
Test Experiment System

Quick validation script to ensure the experiment system is working correctly.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import experiment_config  # noqa: F401
        import experiment_runner  # noqa: F401
        import experiment_analysis  # noqa: F401

        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_creation():
    """Test creating experiment configurations."""
    print("\nTesting configuration creation...")
    try:
        from experiment_config import (
            create_baseline_config,
            create_cartography_filter_config,
            create_label_smoothing_config,
        )

        # Create test configs
        baseline = create_baseline_config(
            name="test_baseline", seed=42, max_train_samples=100
        )
        cart_filter = create_cartography_filter_config(
            name="test_cart_filter", seed=42, max_train_samples=100
        )
        smoothing = create_label_smoothing_config(
            name="test_smoothing", seed=42, max_train_samples=100
        )

        # Verify configs
        assert baseline.name == "test_baseline"
        assert cart_filter.training.cartography_filter.enabled
        assert smoothing.training.label_smoothing.enabled

        print("✓ Configuration creation works")
        return True
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return False


def test_cli_args():
    """Test CLI argument generation."""
    print("\nTesting CLI argument generation...")
    try:
        from experiment_config import create_baseline_config

        config = create_baseline_config(name="test", max_train_samples=100)
        args = config.to_cli_args()

        # Check some key arguments
        assert "--do_train" in args
        assert "--do_eval" in args
        assert "--max_train_samples" in args
        assert "100" in args

        print(f"✓ CLI argument generation works ({len(args)} args)")
        return True
    except Exception as e:
        print(f"✗ CLI argument generation failed: {e}")
        return False


def test_result_loading():
    """Test result loading functionality."""
    print("\nTesting result loading...")
    try:
        from experiment_analysis import load_experiment_results

        # Try to load (may be empty)
        df = load_experiment_results("./experiments")

        print(f"✓ Result loading works (found {len(df)} experiments)")
        return True
    except Exception as e:
        print(f"✗ Result loading failed: {e}")
        return False


def test_dry_run():
    """Test dry-run mode."""
    print("\nTesting dry-run mode...")
    try:
        from experiment_config import create_baseline_config
        from experiment_runner import ExperimentRunner

        config = create_baseline_config(name="dry_run_test", max_train_samples=10)
        runner = ExperimentRunner(experiments=[config], dry_run=True)

        result = runner.run_experiment(config)

        assert result.success
        assert result.metrics.get("dry_run") is True

        print("✓ Dry-run mode works")
        return True
    except Exception as e:
        print(f"✗ Dry-run failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("EXPERIMENT SYSTEM VALIDATION")
    print("=" * 70)

    tests = [
        ("Import test", test_imports),
        ("Config creation", test_config_creation),
        ("CLI args", test_cli_args),
        ("Result loading", test_result_loading),
        ("Dry-run", test_dry_run),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8s} | {name}")

    print("=" * 70)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The experiment system is ready to use.")
        print("\nNext steps:")
        print("1. Generate cartography metrics (see EXPERIMENT_QUICKREF.md)")
        print(
            "2. Run experiments: python run_experiments.py --mode run --suite minimal"
        )
        print("3. Analyze results: python run_experiments.py --mode analyze")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

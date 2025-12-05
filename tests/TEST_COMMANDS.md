# Common Test Commands

## Basic Usage

```bash
# Run all tests
./run_tests.sh

# Or directly with pytest
pytest
```

## Specific Test Files

```bash
# Test filtering functionality
pytest tests/test_dataset_filters.py -v

# Test helpers
pytest tests/test_helpers.py -v

# Test boolean arguments (CLI and W&B sweep compatibility)
pytest tests/test_boolean_arguments.py -v

# Test cartography
pytest tests/test_cartography.py -v

# Test clustering
pytest tests/test_cluster_analysis.py -v

# Test integration (run.py entry points)
pytest tests/test_run_integration.py -v

# Test embeddings and utilities
pytest tests/test_embeddings_and_utils.py -v
```

## Specific Test Classes

```bash
# Test ambiguous question filtering
pytest tests/test_dataset_filters.py::TestAmbiguousQuestionFilter -v

# Test cluster filtering
pytest tests/test_dataset_filters.py::TestClusterFilter -v

# Test boolean argument conversion
pytest tests/test_boolean_arguments.py::TestStrToBool -v

# Test boolean argument parsing (flag vs value style)
pytest tests/test_boolean_arguments.py::TestBooleanArgumentParsing -v

# Test baseline scenarios
pytest tests/test_run_integration.py::TestRunBaseline -v

# Test combined strategies
pytest tests/test_run_integration.py::TestRunCombinedStrategies -v
```

## Specific Tests

```bash
# Test a specific filtering scenario
pytest tests/test_dataset_filters.py::TestAmbiguousQuestionFilter::test_filter_removes_ambiguous_examples -v

# Test flag-style boolean arguments
pytest tests/test_boolean_arguments.py::TestBooleanArgumentParsing::test_flag_style_single_flag -v

# Test W&B sweep-style boolean arguments
pytest tests/test_boolean_arguments.py::TestBooleanArgumentParsing::test_wandb_sweep_style_true -v

# Test cartography loading
pytest tests/test_cartography.py::TestLoadCartographyMetrics::test_load_from_directory -v
```

## Coverage

```bash
# Run with coverage report
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Coverage for specific module
pytest --cov=dataset_filters tests/test_dataset_filters.py
```

## Useful Options

```bash
# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run only failed tests from last run
pytest --lf

# Run tests matching pattern
pytest -k "filter" -v

# Show print statements
pytest -s

# Disable warnings
pytest --disable-warnings

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

## During Development

```bash
# Watch mode - rerun on file changes (requires pytest-watch)
ptw

# Run tests for changed files only
pytest --picked

# Profile slow tests
pytest --durations=10
```

## CI/CD

```bash
# Run all tests with coverage (typical CI setup)
pytest --cov=. --cov-report=xml --cov-report=term -v

# Run with strict mode
pytest --strict-markers --tb=short
```

## Examples

### Test cartography filtering works

```bash
pytest tests/test_dataset_filters.py::TestAmbiguousQuestionFilter -v
```

### Test all integration scenarios

```bash
pytest tests/test_run_integration.py -v
```

### Test specific parameter combination

```bash
pytest tests/test_run_integration.py::TestRunCombinedStrategies::test_cartography_and_cluster_filtering -v
```

### Quick smoke test (fast tests only)

```bash
pytest tests/test_helpers.py tests/test_dataset_filters.py -v
```

### Full test suite with coverage

```bash
./run_tests.sh --cov=. --cov-report=html -v
```

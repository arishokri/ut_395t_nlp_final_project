# Test Suite Summary

## Overview

This comprehensive test suite replaces the `demos/` directory with proper pytest-based tests. The tests cover all major entry points and parameter combinations used in `run.py` and related modules.

## Test Files

### 1. `conftest.py` - Shared Fixtures

- Provides reusable test fixtures for all test files
- Creates sample datasets, metrics, and temporary directories
- Reduces code duplication across tests

### 2. `test_dataset_filters.py` - Filtering Tests

**Coverage:**

- ✅ Ambiguous question filtering (cartography-based)
- ✅ Cluster-based filtering
- ✅ Noise cluster exclusion
- ✅ Multiple cluster exclusion
- ✅ Probability threshold filtering
- ✅ Combined filtering strategies
- ✅ Filter statistics

**Entry Points Tested:**

- `--filter_ambiguous`
- `--filter_clusters`
- `--exclude_noise_cluster`
- `--min_cluster_probability`

### 3. `test_helpers.py` - Helper Function Tests

**Coverage:**

- ✅ Hash ID generation
- ✅ Consistent ID generation
- ✅ Preserving existing IDs
- ✅ QA dataset preparation (train/validation)
- ✅ Ablation studies (q_only, p_only)
- ✅ Metrics computation

**Entry Points Tested:**

- `--ablations` (none, q_only, p_only)
- Dataset preprocessing pipeline

### 4. `test_cartography.py` - Cartography Tests

**Coverage:**

- ✅ Loading cartography metrics from directory/CSV
- ✅ Example categorization (easy/hard/ambiguous)
- ✅ Category-based filtering logic
- ✅ Cartography callback functionality
- ✅ Metrics saving and loading

**Entry Points Tested:**

- `--enable_cartography`
- `--cartography_output_dir`
- `--validation_cartography_output_dir`

### 5. `test_cluster_analysis.py` - Clustering Tests

**Coverage:**

- ✅ Loading cluster assignments
- ✅ Cluster ID validation
- ✅ Noise cluster detection
- ✅ Probability column handling
- ✅ Cluster statistics

**Entry Points Tested:**

- `--cluster_assignments_path`
- `--validation_cluster_assignments_path`

### 6. `test_run_integration.py` - Integration Tests

**Coverage:**

- ✅ Baseline training arguments
- ✅ Cartography-enabled training
- ✅ Cluster filtering
- ✅ Label smoothing
- ✅ Soft weighting
- ✅ Combined strategies
- ✅ Ablation studies
- ✅ Validation filtering

**Entry Points Tested:**

- `--do_train`
- `--enable_cartography`
- `--filter_ambiguous`
- `--filter_clusters`
- `--use_label_smoothing`
- `--smoothing_factor`
- `--use_soft_weighting`
- `--weight_clip_min`
- `--weight_clip_max`
- `--filter_validation`
- `--train_split`
- `--ablations`

### 7. `test_embeddings_and_utils.py` - Utilities Tests

**Coverage:**

- ✅ Embedding file format validation
- ✅ Metadata validation
- ✅ Embedding normalization
- ✅ Rule-based filtering
- ✅ Dataset analysis
- ✅ Training dynamics structure
- ✅ Output format validation

## Parameter Coverage Matrix

| Parameter                               | Tested | Test File                                         |
| --------------------------------------- | ------ | ------------------------------------------------- |
| `--model`                               | ✅     | test_run_integration.py                           |
| `--dataset`                             | ✅     | test_run_integration.py                           |
| `--max_length`                          | ✅     | test_helpers.py                                   |
| `--max_train_samples`                   | ✅     | test_run_integration.py                           |
| `--max_eval_samples`                    | ✅     | test_run_integration.py                           |
| `--ablations`                           | ✅     | test_run_integration.py, test_helpers.py          |
| `--enable_cartography`                  | ✅     | test_run_integration.py, test_cartography.py      |
| `--cartography_output_dir`              | ✅     | test_cartography.py, test_run_integration.py      |
| `--filter_ambiguous`                    | ✅     | test_dataset_filters.py, test_run_integration.py  |
| `--filter_clusters`                     | ✅     | test_dataset_filters.py, test_run_integration.py  |
| `--cluster_assignments_path`            | ✅     | test_cluster_analysis.py, test_run_integration.py |
| `--exclude_noise_cluster`               | ✅     | test_dataset_filters.py, test_run_integration.py  |
| `--min_cluster_probability`             | ✅     | test_dataset_filters.py, test_run_integration.py  |
| `--use_label_smoothing`                 | ✅     | test_run_integration.py                           |
| `--smoothing_factor`                    | ✅     | test_run_integration.py                           |
| `--use_soft_weighting`                  | ✅     | test_run_integration.py                           |
| `--weight_clip_min`                     | ✅     | test_run_integration.py                           |
| `--weight_clip_max`                     | ✅     | test_run_integration.py                           |
| `--train_split`                         | ✅     | test_run_integration.py                           |
| `--filter_validation`                   | ✅     | test_run_integration.py                           |
| `--validation_cartography_output_dir`   | ✅     | test_run_integration.py                           |
| `--validation_cluster_assignments_path` | ✅     | test_run_integration.py                           |

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_dataset_filters.py

# Run specific test class
pytest tests/test_dataset_filters.py::TestAmbiguousQuestionFilter

# Run with coverage report
pytest --cov=. --cov-report=html
```

### During Development

```bash
# Run tests related to your changes
pytest tests/test_dataset_filters.py -v

# Run and stop on first failure
pytest -x

# Run only failed tests from last run
pytest --lf

# Run tests matching a pattern
pytest -k "filter"
```

## Test Strategy

### Unit Tests

- Test individual functions and classes in isolation
- Use fixtures to provide consistent test data
- Fast execution (<1 second per test)

### Integration Tests

- Test argument parsing and validation
- Verify combinations of parameters work together
- Don't require full training (use small samples)
- May timeout if environment issues occur

### Fixtures

- All tests use shared fixtures from `conftest.py`
- Fixtures provide:
  - Sample datasets (5 examples)
  - Temporary directories (auto-cleaned)
  - Sample metrics and assignments
  - Mock model paths

## Benefits Over demos/

1. **Automated**: Can be run automatically in CI/CD
2. **Comprehensive**: Tests all parameter combinations
3. **Fast**: Most tests run in milliseconds
4. **Maintainable**: Clear organization and fixtures
5. **Preventive**: Catches breaking changes before deployment
6. **Documented**: Tests serve as usage examples

## Migration from demos/

The old demo files tested specific scenarios:

- `cartography_filtering_demo.py` → `test_dataset_filters.py` + `test_cartography.py`
- `cluster_filtering_demo.py` → `test_dataset_filters.py` + `test_cluster_analysis.py`
- `unified_analysis_demo.py` → `test_run_integration.py`
- `test_offsets.py` → `test_helpers.py`

All functionality is now covered with proper assertions and error handling.

## Continuous Integration

These tests are designed to work in CI environments:

- No GPU required for most tests
- Use small sample data
- Auto-generate temporary files
- Clean up after themselves
- Can skip slow/resource-intensive tests with markers

## Next Steps

1. Run tests to ensure current codebase works: `pytest -v`
2. Add tests as you refactor: Write test first, then refactor
3. Check coverage: `pytest --cov=. --cov-report=html`
4. Fix any failing tests before merging changes

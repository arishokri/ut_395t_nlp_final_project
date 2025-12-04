# Test Suite

This directory contains comprehensive tests for the QA Cartography project.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and test configuration
├── test_dataset_filters.py        # Tests for filtering strategies
├── test_helpers.py                 # Tests for helper functions
├── test_cartography.py            # Tests for cartography functionality
├── test_cluster_analysis.py       # Tests for clustering
└── test_run_integration.py        # Integration tests for run.py
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_dataset_filters.py
```

### Run specific test class

```bash
pytest tests/test_dataset_filters.py::TestAmbiguousQuestionFilter
```

### Run specific test

```bash
pytest tests/test_dataset_filters.py::TestAmbiguousQuestionFilter::test_filter_removes_ambiguous_examples
```

### Run with coverage

```bash
pytest --cov=. --cov-report=html
```

### Run only fast tests (skip slow integration tests)

```bash
pytest -m "not slow"
```

## Test Coverage

### Unit Tests

- **test_dataset_filters.py**: Tests all filtering strategies

  - Ambiguous question filtering
  - Cluster-based filtering
  - Combined filtering
  - Filter statistics

- **test_helpers.py**: Tests helper utilities

  - Hash ID generation
  - QA dataset preparation
  - Metrics computation
  - Ablation studies

- **test_cartography.py**: Tests cartography functionality

  - Loading cartography metrics
  - Example categorization (easy/hard/ambiguous)
  - Cartography callback

- **test_cluster_analysis.py**: Tests clustering
  - Loading cluster assignments
  - Cluster probability handling
  - Noise cluster identification

### Integration Tests

- **test_run_integration.py**: Tests run.py entry points
  - Baseline training
  - Cartography-enabled training
  - Cluster filtering
  - Label smoothing
  - Soft weighting
  - Combined strategies
  - Ablation studies
  - Validation filtering

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for test outputs
- `sample_qa_dataset`: Small QA dataset (5 examples)
- `sample_cartography_metrics`: Sample cartography metrics CSV
- `sample_cluster_assignments`: Sample cluster assignments CSV
- `sample_embeddings`: Sample embeddings numpy file
- `mock_model_path`: Model identifier for testing
- `training_args_dict`: Basic training arguments

## Adding New Tests

1. Create test file following naming convention: `test_<module>.py`
2. Use fixtures from `conftest.py` or create new ones
3. Organize tests into classes using `Test<Feature>` naming
4. Name test methods with descriptive names: `test_<what_it_does>`

Example:

```python
class TestMyFeature:
    def test_feature_works_correctly(self, sample_qa_dataset):
        # Arrange
        dataset = sample_qa_dataset

        # Act
        result = my_function(dataset)

        # Assert
        assert len(result) > 0
```

## Notes

- Integration tests may require GPU/model downloads and are more resource-intensive
- Some tests validate argument parsing without full training execution
- Tests use small datasets to run quickly
- Temporary directories are automatically cleaned up after tests

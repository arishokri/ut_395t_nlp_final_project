#!/bin/bash
# Quick test runner script
# Usage: ./run_tests.sh [options]

set -e

echo "========================================================================"
echo "QA Cartography Experiments - Test Suite"
echo "========================================================================"
echo ""

# Parse arguments
TEST_ARGS="${@:-}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if pytest is installed
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "pytest not found. Installing..."
    pip install pytest pytest-cov
fi

echo ""
echo "Running tests..."
echo "------------------------------------------------------------------------"

# Run tests based on arguments or default to all tests
if [ -z "$TEST_ARGS" ]; then
    # Default: run all tests with verbose output
    pytest tests/ -v
else
    # Run with user-provided arguments
    pytest "$TEST_ARGS"
fi

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $EXIT_CODE)"
fi
echo "========================================================================"
echo ""

exit $EXIT_CODE

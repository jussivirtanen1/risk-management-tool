#!/bin/bash
set -e

# Function to run tests
run_tests() {
    echo "============================================"
    echo "Starting test execution in test environment"
    echo "============================================"
    
    pytest tests/ -v
    TEST_RESULT=$?
    
    echo "============================================"
    if [ $TEST_RESULT -eq 0 ]; then
        echo "✅ All tests passed successfully"
    else
        echo "❌ Tests failed with exit code: $TEST_RESULT"
    fi
    echo "============================================"
    
    return $TEST_RESULT
}

# Main execution
if [ "$ENV" = "_test" ]; then
    run_tests
    exit $?
else
    echo "Starting application in ${ENV:-default} environment"
    exec python main.py
fi 
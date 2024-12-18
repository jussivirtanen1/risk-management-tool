#!/bin/bash
set -e

# Function to run tests
run_tests() {
    echo "Running tests..."
    pytest tests/ -v
    return $?
}

# Function to start the application
start_app() {
    echo "Starting the application..."
    exec python main.py
}

# Main execution
if [ "$ENV" = "_test" ]; then
    echo "Test environment detected."
    run_tests
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo "Tests passed successfully."
        start_app
    else
        echo "Tests failed. Application will not start."
        exit $TEST_RESULT
    fi
else
    echo "Environment: ${ENV:-default}"
    start_app
fi 
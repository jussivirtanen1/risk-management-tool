#!/bin/bash
set -e  # Exit on error

if [ "$1" = "test" ]; then
    echo "Running tests..."
    pytest tests/test_portfolio_analyzer.py -v -s
elif [ "$1" = "app" ]; then
    echo "Starting application with environment: $ENV"
    python main.py
else
    exec "$@"
fi
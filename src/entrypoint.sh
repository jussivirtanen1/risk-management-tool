#!/bin/bash
set -e  # Exit on error

if [ "$1" = "test" ]; then
    echo "Running tests..."
    pytest tests/ -v
elif [ "$1" = "app" ]; then
    echo "Starting application with environment: $DB_PARAM"
    python main.py
else
    exec "$@"
fi
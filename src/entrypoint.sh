#!/bin/bash
set -e

# Check if we're running pytest
if [[ "$*" == *"pytest"* ]]; then
    # Default to test environment for pytest if DB_PARAM not set
    if [ -z "$DB_PARAM" ]; then
        echo "Running tests - defaulting DB_PARAM to test"
        export DB_PARAM="test"
    fi
else
    # For non-test runs, DB_PARAM must be explicitly set
    if [ -z "$DB_PARAM" ]; then
        echo "Error: DB_PARAM must be set"
        exit 1
    fi
fi

# Remove any leading underscore if present
DB_PARAM=${DB_PARAM#_}

# Validate required environment variables
if [ -z "$DB_USER" ]; then
    echo "Error: DB_USER must be set"
    exit 1
fi

if [ -z "$DB_PASSWORD" ]; then
    echo "Error: DB_PASSWORD must be set"
    exit 1
fi

echo "Starting application with environment: $DB_PARAM"
exec python main.py
#!/bin/bash
set -e

# Validate DB_PARAM
if [ -z "$DB_PARAM" ]; then
    echo "Warning: DB_PARAM is not set, defaulting to test environment"
    export DB_PARAM="_test"
elif [ "$DB_PARAM" != "_prod" ] && [ "$DB_PARAM" != "_test" ]; then
    echo "Error: DB_PARAM must be either '_prod' or '_test'"
    exit 1
fi

echo "Starting application with database parameter: $DB_PARAM"
exec python main.py 
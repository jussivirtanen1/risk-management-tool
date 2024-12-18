#!/bin/bash
set -e

echo "Starting application with DB_PARAM=${DB_PARAM:-default}"
exec python main.py 
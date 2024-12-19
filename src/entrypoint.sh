   #!/bin/bash
   set -e

   # Check if we're running pytest
   if [[ "$1" == "pytest" ]]; then
       # Default to test environment for pytest if DB_PARAM not set
       if [ -z "$DB_PARAM" ]; then
           echo "Running tests - defaulting DB_PARAM to test"
           export DB_PARAM="test"
       fi
       # Run pytest and exit
       echo "Running pytest..."
       exec "$@"
       exit 0
   else
       # For non-test runs, DB_PARAM must be explicitly set
       if [ -z "$DB_PARAM" ]; then
           echo "Error: DB_PARAM must be set"
           exit 1
       fi
   fi

   # Remove any leading underscore if present
   DB_PARAM=${DB_PARAM#_}

   echo "Starting application with environment: $DB_PARAM"
   exec python main.py
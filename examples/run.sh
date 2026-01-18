#!/bin/bash

if [ -f "$1" ] && [[ "$1" == *.py ]]; then
    # If a python file is passed as an argument, run it.
    file="$1"
    echo "Running $file"
    cd "$(dirname "$file")"
    python "$(basename "$file")"
    exit 0
elif [ -d "$1" ]; then
    # if a folder is passed as an argument, run all python scripts within it.
    find "$1" -type f -name "*.py" | while read -r file; do
        echo "Running $file"
        cd "$(dirname "$file")"
        python "$(basename "$file")"
    done
    exit 0
else
    echo "$1 is not a Python file or directory."
    exit 1
fi
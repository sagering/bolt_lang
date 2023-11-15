#!/bin/bash

# Specify the directory to iterate through
directory="tests"

# Check if the directory exists
if [ -d "$directory" ]; then
    # Iterate through each file in the specified directory
    for file in "$directory"/*; do
        # Check if it's a regular file
        if [ -f "$file" ]; then
            # Strip the file extension
            filename=$(basename "$file")
            filename_no_ext="${filename%.*}"

            # Run test
            ./run_single_test.sh "$filename_no_ext"
        fi
    done
else
    echo "Directory not found: $directory"
fi


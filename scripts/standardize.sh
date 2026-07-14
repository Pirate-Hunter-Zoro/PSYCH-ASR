#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

if [ $# -lt 1 ]; then
    echo "Missing input .m4a file" >&2 # Print to stderr
    exit 1
fi

input="$1"
output="${input%.m4a}.wav" # Strip .m4a and replace with .wav for output file

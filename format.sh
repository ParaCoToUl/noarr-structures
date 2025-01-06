#!/bin/bash

# This script runs clang-format on all files in include/ to ensure consistent formatting.

set -e

find include/ \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
	-exec clang-format -i -style=file {} +

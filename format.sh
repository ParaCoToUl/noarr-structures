#!/bin/bash

# This script runs clang-format on all files in include/ to ensure consistent formatting.

set -euo pipefail

print_usage() {
	echo "Usage: $0 [--check|--help]"
	echo ""
	echo "Options:"
	echo "  --check    Check if files are properly formatted without making changes."
	echo "  --help     Show this help message."
}

if ! command -v clang-format &> /dev/null; then
	echo "clang-format could not be found. Please install it to use this script." >&2
	exit 1
fi

if [ ! -f .clang-format ]; then
	echo ".clang-format file not found in the current directory. Please create one to specify formatting style." >&2
	exit 1
fi

if [ $# -eq 1 ] && [ "$1" == "--check" ]; then
	echo "Checking code format..."
	if ! find include/ \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
		-exec clang-format --dry-run --Werror -style=file {} +; then
		echo "Code format check failed. Please run ./format.sh to format the code." >&2
		exit 1
	else
		echo "All files are properly formatted." >&2
		exit 0
	fi
fi

if [ $# -eq 1 ] && [ "$1" == "--help" ]; then
	print_usage
	exit 0
fi

if [ $# -gt 0 ]; then
	print_usage >&2
	exit 1
fi

find include/ \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
	-exec clang-format -i -style=file {} +

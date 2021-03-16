PYTHON_VERSION=3.6
LIB_NAME=foo

.PHONY: compile run

compile:
	g++ -std=c++14 -Wall -Wextra -pedantic -Werror \
		src/main.cpp \
		-o bin/main

run: compile
	bin/main

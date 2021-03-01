PYTHON_VERSION=3.6
LIB_NAME=foo

library:
	g++ -std=c++14 -Wall -Wextra -pedantic -Werror \
		-shared -fPIC \
		src/algorithm.cpp \
		-o bin/lib$(LIB_NAME).so

python: library
	g++ -O3 -Wall -Werror -shared -std=c++14 -fPIC \
	`python3 -m pybind11 --includes` \
    -I /usr/include/python$(PYTHON_VERSION) -I ./src \
    ./python/binding.cpp \
    -o ./bin/$(LIB_NAME)`python$(PYTHON_VERSION)-config --extension-suffix` \
    -L./bin -l$(LIB_NAME) -Wl,-rpath,./bin

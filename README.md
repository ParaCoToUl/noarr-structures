![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20test%20ubuntu-latest%20-%20clang/badge.svg)
![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20test%20ubuntu-latest%20-%20gcc/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20test%20macosl/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20test%20Win/badge.svg)

# Noarr Structures

Header-only library that facilitates creation of many data structures and provides a layout-agnostic way of accessing the values stored in them.

**Data layout modeling:**

> Note that we model data seperately because TODO

The following example demonstrates the layout-agnostic method of accessing the values:

```cpp
// the following two structures both describe a two-dimensional continuous array (matrix)

// creates a 20x30 two-dimensional array
noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>> foo;

// creates the same structure with A different physical layout
noarr::array<'y', 30, noarr::array<'x', 20, noarr::scalar<int>>> bar;

// getting the offset of the value at (x = 5; y = 10):
foo | noarr::offset<'x', 'y'>(5, 10);
bar | noarr::offset<'x', 'y'>(5, 10);
```

**Data accessing:**

The library then provides `noarr::bag`, a basic data structure that allocates enough memory for the desired structure and provides the `at` method (used like: `at<'x', 'y'>(5, 10)`), which returns a reference to a value stored in the allocated memory. You can transpose matrix like this:

```cpp
template<typename Structure>
void matrix_transpose(noarr::bag<Structure>& matrix1) {
	for (int i = 0; i < matrix1.template get_length<'x'>(); i++)
		for (int j = i; j < matrix1.template get_length<'y'>(); j++)
			std::swap(matrix1.at<'x', 'y'>(i, j), matrix1.at<'x', 'y'>(j, i));
}
```

See [examples/matrix](examples/matrix "matrix example") for a demo.

## Using the library

Noarr structures is a header-only library, so only include path need to be added. The include path should point to the `/include` folder of this repository.

```cmake
# the CMake line that adds the include directory
target_include_directories(<my-app> PUBLIC <cloned-repo-path>/include)
```

The library requires C++ 17.


## Running tests and examples

Enter the desired folder ([examples/matrix](examples/matrix "matrix example")). In the terminal (linux bash, windows cygwin or gitbash) run the following commands:

```sh
# generates build files for your platform
cmake .

# builds the project using previously generated build files
cmake --build .

# run the built executable
# (this step differs by platform, this example is for linux)
./matrix || ./test-runner
```

# Noarr Structures

## Noarr tests

![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20test%20ubuntu-latest%20-%20clang/badge.svg)
![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20test%20ubuntu-latest%20-%20gcc/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20test%20macosl/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20test%20Win/badge.svg)

## Introduction

Noarr structures is a header-only library that facilitates creation of many data structures and provides a layout-agnostic way of accessing the values stored in them.

It is a free software and distributed using the MIT [license](LICENSE).

The library provides its low-level parts, *structures* and *functions*, as simple and easy-to-use building blocks, that can be used alongside with some memory management to easily create data structures that provide layout-agnostic access, and nigh-interplatform type checking and serialization.

If you are not interested in creating a data structure yourself, you can use `noarr::bag` which uses the standard c++ memory management. (see: *`noarr::bag`: flexible data structure*)

## Data layout modeling

This section contains the low-level building blocks the library consists of, these are in background in the higher-level concepts like `noarr::bag` which is a simple data structure described in the next section.

The main feature of the library are *structures* which describe a layout of the data, they do not contain the data themselves which makes them a good tool for creating such data structures with any memory management specifics.

These structures are then combined with *functions* and together they create an expressive and flexible system capable of describing a vast amount of layouts and offering various ways of using them to access data.

The following example demonstrates the layout-agnostic method of accessing the values using structures and functions defined in the library:

```cpp
// the following two structures both describe a two-dimensional 
	// continuous array (matrix)

// describes a layout of 20x30 two-dimensional array
noarr::array<'x', 20, noarr::array<'y', 30, 
	noarr::scalar<int>>> foo;

// describes a similar logical layout with switched dimensions 
	// in the physical layout
noarr::array<'y', 30, noarr::array<'x', 20, 
	noarr::scalar<int>>> bar;

// getting the offset of the value at (x = 5; y = 10):
foo | noarr::offset<'x', 'y'>(5, 10);
bar | noarr::offset<'x', 'y'>(5, 10);
```

## `noarr::bag`: flexible data structure

The data structure `noarr::bag<Structure>` uses the standard c++ memory management to contain some data described by `Structure` (`Structure` usually consists of nested `noarr::array`s or `noarr::vector`s). Then the data are accessed by the method `at<Dimensions>(indices)` which takes the names of dimensions as type parameters and their corresponding indices as function parameters, for example: `at<'x', 'y'>(5, 10)` (this allows for layout-agnostic data accessing demonstrated in [examples/matrix](examples/matrix "matrix example"), the main part of the demonstration is described in the following snippets in a very simplified form).

The following snippet shows how we define a matrix structure and then we use `noarr::bag` to create a data structure called `matrix1` consisting of the matrix structure with certain dimensions.

```cpp

// defines the structure of the matrix, rows are the 'x' dimension 
	// and columns are the 'y' dimension
// physically, the layout is an contiguous array of rows
noarr::vector<'y', noarr::vector<'x', 
	noarr::scalar<int>>> matrix_structure;

// defining size of the matrix
auto sized_matrix_structure = matrix_structure 
	| noarr::set_length<'x'>(WIDTH) 
	| noarr::set_length<'y'>(HEIGHT);

// data allocation
auto matrix = noarr::bag(sized_matrix_structure);
```

The following snippet then shows how we would transpose the values of the matrix using the `at` method:

```cpp
for (std::size_t i = 0; i < matrix.get_length<'x'>(); i++)
	for (std::size_t j = i; j < matrix.get_length<'y'>(); j++)
		std::swap(
			matrix.at<'x', 'y'>(i, j), 
			matrix.at<'x', 'y'>(j, i));
```

In this snippet the actual physical layout of the matrix is not relevant to the way it is accessed. If the data were stored, for example, in an (contiguous) array of columns, it could still be accessed the same way. This contrasts with traditional C/C++ data structures.

## Using the library

Noarr structures is a header-only library, so only include path need to be added. The include path should point to the `/include` folder of this repository.

```cmake
# the CMake line that adds the include directory
target_include_directories(<my-app> PUBLIC <cloned-repo-path>/include)
```

The library requires C++ 17.

## Examples

Examples can be found at [examples/matrix](examples/matrix "matrix example").

### Matrix example tests

![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20matrix%20example%20test%20ubuntu-latest%20-%20clang/badge.svg)
![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20matrix%20example%20test%20ubuntu-latest%20-%20gcc/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20matrix%20example%20test%20macosl/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr-structures/workflows/Noarr%20matrix%20example%20test%20Win/badge.svg)

## Running tests

Make sure you are in the root folder. In the terminal (linux bash, windows cygwin or gitbash) run the following commands:

```sh
# creates the build directory
cmake -E make_directory build

# enters the build directory
cd build

# configures the build environment
cmake ..

# builds the project according to the configuration
cmake --build .
```

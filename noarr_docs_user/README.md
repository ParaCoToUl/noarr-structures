# User documentation for Noarr

<a name="data-modelling"></a>
## Data modelling

Data modelling is the process of describing the structure of your data, so that an algorithm can be written to processes the data. Noarr lets you model your data in an abstract, multidimensional space, abstracting away any underlying physical structure.

Noarr framework distinguishes two types of multidimensional data - uniform and jagged.

**Jagged data** can be thought of as a vector of vectors, each having different size. This means the dimensions of such data need to be stored within the data itself, requiring the use of pointers and making processing of such data inefficient. Noarr supports this type of data only at the highest abstraction levels of your data model.

**Uniform data** can be though of as a multidimensional cube of values. It is like a vector of same-sized vectors, but it also supports tuples and other structures. This lets us store the dimensions separately from the data, letting us freely change the order of specification of dimensions - completely separating the physical data layout from the data model.


<a name="data-modelling-in-noarr"></a>
### Data modelling in Noarr

*Noarr structures* was designed to support uniform data. Uniform data has the advantage of occupying one continuous array of memory. When working with it, you work with three objects:

1. **Structure:** A small, tree-like object, that represents the structure of the data. It does not contain the data itself, nor a pointer to the data. It can be thought of as a function that maps indices to memory offsets (in bytes). It stores information, such as data dimensions and tuple types.
2. **Data:** A continuous block of bytes that contains the actual data. Its structure is defined by a corresponding *Structure* object.
3. **Bag:** Wrapper object, which combines *structure* and *data* together.

> Note: in the case of jagged data, you can use *Noarr pipelines* without *Noarr structures*. The architecture of the GPU is designed for uniform data mainly, so it should fit most common cases. Also note, that you can also use several *Noarr structures* in your program.

#### Creating a structure

First, we have to `#include` the `noarr` library:

```cpp
#include "noarr/structures_extended.hpp"
```

To represent a list of floats, you create the following *structure* object:

```cpp
noarr::vector<'i', noarr::scalar<float>> my_structure;
```

The only dimension of this *structure* has the label `i` and it has to be specified in order to access individual scalar values. But currently the structure has no size, we need to make room for 10 items:

```cpp
auto my_structure_of_ten = my_structure | noarr::set_length<'i'>(10);
```

A *structure* object is immutable. The `|` operator (the pipe) is used to create modified variants of *structures*. You can chain such operations to arrive at the structure that represents your data.

> The pipe operator is the preferred way to query or modify structures, as it automatically locates the proper sub-structure with the given dimension label (in compile time).

The reason we specify the size later is that it allows us to decouple the *structure* structure from the resizing action. The resizing action specifies a dimension label `i` and it does not care, where that dimension is inside the *structure*.

<a name="wrapper"></a>
#### Wrapper
It is possible to use `.` (dot) instead of `|` (pipe), but you have to use `noarr::wrapper` first.

```cpp
// artificially complicated example
auto piped = my_structure_of_ten | noarr::set_length<'i'>(5) | noarr::set_length<'i'>(10);
// now version with wrapper
auto doted = noarr::wrap(my_structure_of_ten).set_length<'i'>(5).set_length<'i'>(10);
```

#### Allocating and accessing *data* and *bag*

Now that we have a structure defined, we can create a bag to store the data. Bag allocates *data* buffer automatically:

```cpp
// we will create a bag
auto bag = noarr::make_bag(my_structure_of_ten);
```


Now, with a *data* that holds the values, we can access these values by computing their offset in the *bag*:

```cpp
// get the reference (we will get 5-th element)
float& value_ref = bag.structure().get_at<'i'>(bag.data(), 5);

// now use the reference to access the value
value_ref = 42;
```

As discussed earlier, there is a good reason to separate *structure* and *data*. But in the case of *bag*, there is the following shortcut:

```cpp
bag.at<'i'>(5) = 42;
```

<a name="changing-data-layouts"></a>
#### Changing data layout (*structure*)

Now we want to change the data layout. Noarr needs to know the structure at compile time (for performance). So the right approach is to template all functions and then select between compiled versions. We define different structures like this:

```cpp
// layout declaration
using matrix_rows = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
using matrix_columns = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
```

We will create a templated matrix. And also set size at runtime like this:

```cpp
// function which does some logic templated by different structures
template<typename Structure>
void matrix_demo(int size) {
	// dot version
	// note template keyword, it is there because the whole function is layout templated
	auto n1 = noarr::make_bag(noarr::wrap(Structure()).template set_length<'x'>(size).template set_length<'y'>(size));
	// pipe version (both are valid syntax and produce the same result)
	auto n2 = noarr::make_bag(Structure() | noarr::set_length<'x'>(size) | noarr::set_length<'y'>(size));
}
```

We set the size at runtime because size can be any int.

We can call at runtime different templated layouts.

```cpp
void main() {
	...
	// we select the layout in runtime
	if (layout == "rows")
		matrix_demo<matrix_rows>(size);
	else if (layout == "columns")
		matrix_demo<matrix_columns>(size);
	...
}
```

<a name="supported-layouts"></a>
#### Our supported layouts (*structures*)
##### Containers

Noarr supports `vector` and `array`. Our library is designed to be easily extendable. We have implemented a 2D z-curve in  our matrix example [z-curve implementation example](../examples/matrix/z_curve.hpp "z-curve implementation example"). Basic declarations look like this:

```cpp
noarr::vector<'i', noarr::scalar<float>> my_vector;
noarr::array<'i', 10, noarr::scalar<float>> my_array;
```

##### Scalars

Noarr supports all scalars, for example: `bool`, `int`, `char`, `float`, `double`, `long`, `std::size_t`... 

You can read about supported scalars in detail in [technical documentation](../noarr_docs_tech/README.md "technical documentation").

##### Tuples

Here are some of the valid tuple declarations:

```cpp
noarr::tuple<'t', noarr::scalar<int>, noarr::scalar<float>> t;
noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'x', noarr::scalar<int>>> t2;
noarr::tuple<'t', noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>>, 
	noarr::vector<'x', noarr::array<'y', 20, noarr::scalar<int>>>> t3;
```

We will work with `tuple`s like this:

```cpp
// tuple declaration
noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::array<'x', 20, noarr::scalar<int>>> tuple;
// we will create a bag
auto tuple_bag = noarr::make_bag(tuple);
// we have to use noarr::literals namespace to be able to index tuples
// we can put this at the beginning of the file
using namespace noarr::literals;
// we index tuple like this
// note that we fix multiple dimentions at one
float& value = tuple_bag.at<'t', 'x'>(0_idx, 1);
```

<a name="full-list-of-functions"></a>
#### Full list of functions

  - `compose`: function composition (honoring the left-associative notation)
  - `set_length`: changes the length (number of indices) of arrays and vectors
  - `get_length`: gets the length (number of indices) of a structure
  - `get_size`: returns size of the structure in bytes
  - `fix`: fixes an index in a structure
  - `get_offset`: retrieves offset of a substructure 
  - `offset`: retrieves offset of a value in a structure with no dimensions (or in a structure with all dimensions being fixed), allows for ad-hoc fixing of dimensions
  - `get_at`: returns a reference to a value in a given blob the offset of which is specified by a dimensionless (same as `offset`) structure, allows for ad-hoc fixing of dimensions
  - `contain`: tuple-like struct, and a structured layout, that facilitates creation of new structures and functions

You can read about supported functions in detail in [Noarr structures](../noarr/include/noarr/structures/README.md "Noarr structures").


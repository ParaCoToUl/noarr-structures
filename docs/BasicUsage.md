# Basic Usage

First, we have to `#include` the `noarr` library:

```cpp
#include "noarr/structures_extended.hpp"
```

To represent a list of floats, you create the following *structure* object:

```cpp
noarr::vector<'i', noarr::scalar<float>> my_structure;
// ~or~
auto my_structure = noarr::scalar<float>() ^ noarr::vector<'i'>();
```

The above notations are equivalent, but for some structures (described below), only the latter is available. The `^` operator (originally xor in C++) stands for exponential type.
It takes the structure on the left-hand-side and wraps it in the structure prototype on the right-hand-side.

The only dimension of this *structure* has the label `i` and it has to be specified in order to access individual scalar values.


## Lengths

Currently, the structure has no size, we need to make room for 10 items:

```cpp
auto my_structure_of_ten = my_structure 
	^ noarr::set_length<'i'>(10);
```

Here we use wrap the structure in `set_length`. Like `vector`, `set_length` is also a structure. This new structure is returned by the operator while the original structure is left unchanged.

After the length is set, it can be queried using function `get_length`. Unlike `set_length`, `get_length` is not a structure prototype and the result is obviously not a structure.
We use `|` (pipe) for function application to distinguish it from `^`:

```cpp
std::size_t ten = my_structure_of_ten | noarr::get_length<'i'>();
```


## Functions

There are other functions.

### get_size

`get_size` returns the size in bytes (it does not take any dimension as a parameter, since it considers all dimensions):

```cpp
std::size_t forty = my_structure_of_ten | noarr::get_size();
```

### offset

`offset` retrieves offset of a scalar value, allowing for ad-hoc fixing of dimensions:

```cpp
std::size_t eight = my_structure_of_ten | noarr::offset<'i'>(2);
```

### get_at

`get_at` returns a reference to a value in a given blob the offset of which is specified by a dimensionless (same as `offset`) structure, allowing for ad-hoc fixing of dimensions:

```cpp
void *data = ...;
float &last_elem = my_structure_of_ten | noarr::get_at<'i'>(data, 9);
```


## Bag

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

### Indexing a bag

A bag can be indexed using a [state](State.md). The above can be rewritten as:

```cpp
auto state = noarr::idx<'i'>(5);
bag[state] = 42;
```

<!-- TODO elaborate -->

### Bag types

There are various types of `bag`s:

- vector bag

  - created by `make_vector_bag(structure)`
  - the underlying data blob is implemented via a `std::vector`
  - the bag destroys the data blob in its destructor

- unique bag (the default type)

  - created by `make_unique_bag(structure)` or `make_bag(structure)`
  - the underlying data blob is implemented via a `std::unique_ptr`
  - the bag destroys the data blob in its destructor

- observer bag

  - created by `make_bag(structure, char_carray)`
  - the underlying data blob is implemented via `char *`
  - destruction of bag does not affect the data blob


## Changing data layout (*structure*)

Now we want to change the data layout. Noarr needs to know the structure at compile time (for performance). So the right approach is to template all functions and then select between compiled versions. We define different structures like this:

```cpp
// layout declaration
using matrix_rows = noarr::vector<'y', 
	noarr::vector<'x', noarr::scalar<int>>>;
using matrix_columns = noarr::vector<'x', 
	noarr::vector<'y', noarr::scalar<int>>>;
```

We will create a templated matrix. And also set size at runtime like this:

```cpp
// function which does some logic templated by different structures
template<typename Structure>
void matrix_demo(int size) {
	auto n2 = noarr::make_bag(Structure() 
		^ noarr::set_length<'x'>(size) 
		^ noarr::set_length<'y'>(size));
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

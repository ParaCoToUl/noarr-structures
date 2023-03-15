# Basic Usage

First, we have to `#include` the `noarr` library:

```cpp
#include <noarr/structures_extended.hpp>
```

We start by modeling the layout. For example, either of the following two equivalent definitions describes a list of floats (indexed by some `'i'`):

```cpp
noarr::vector<'i', noarr::scalar<float>> my_structure;
// ~ or ~
auto my_structure = noarr::scalar<float>() ^ noarr::vector<'i'>();
```

Note that we have *not* created any "container" yet. `my_structure` is just a description of one. We call these descriptions [structures](Glossary.md#structure).
(Think *the structure of the data*, not *a data structure*.)

The `^` operator (originally xor in C++) takes the left-hand side and wraps it in the right-hand side.
The right-hand side is not exactly a [structure](Glossary.md#structure). It is missing an important part.
Such objects are called [proto-structures](Glossary.md#proto-structure) in noarr, and being used in `^` is their primary purpose.
[Defining Structures](DefiningStructures.md) describes the `^` operator and its meaning in full detail.

Why two syntaxes? While the former is more familiar to a C++ programmer (and is closer to how the structure is represented),
it quickly becomes unusable when one starts to nest structures and manipulate the layouts. For now, we will stick with the latter syntax.

And what does `scalar` do? In both syntaxes, the `vector` structure needs another [structure](Glossary.md#structure) to specify the element type.
We could use another vector for the element to create a 2D layout (*), but here we want the elements to be just simple float values.
And [`scalar<T>`](structs/scalar.md) is exactly that: a structure that describes a layout consisting of just one `T` value.

(*): and even so, we would need an element type for the inner vector; it cannot be vecturtles all the way down.

```cpp
auto my_matrix = noarr::scalar<float>() ^ noarr::vector<'i'>() ^ noarr::vector<'j'>();
```


## Lengths

Unlike `std::vector`, `noarr::vector` cannot grow. Its [length](Glossary.md#length) must be set before it is used:

```cpp
auto my_structure_of_ten = my_structure ^ noarr::set_length<'i'>(10);
auto my_matrix_3x3 = my_matrix ^ noarr::set_length<'i', 'j'>(3, 3);
```

Like `vector`, [`set_length`](structs/set_length.md) is a [proto-structure](Glossary.md#proto-structure).
It does not add any dimensions or elements, it just communicates to the underlying `vector` what it should use for its length.
Note that the original structure is not modified in-place. Instead, a new structure is returned.

You can also use [`sized_vector`](structs/sized_vector.md) or [`array`](structs/array.md), which are shortcuts for a plain `vector` combined with a `set_length`.
On the other hand, it is possible [to not include](other/SeparateLengths.md) the length in the structure at all.

Once the length is set, it cannot be updated using another `^ noarr::set_length`. On the other hand, it can be queried using `get_length`:

```cpp
std::size_t ten = my_structure_of_ten | noarr::get_length<'i'>();
std::size_t three = my_matrix_3x3 | noarr::get_length<'i'>(); // same with 'j'
```

Notice the difference in the operator used (`|` and not `^`). That is because `get_length` is *not* a [proto-structure](Glossary.md#proto-structure).
It is a noarr function and `|` calls the function on the structure. You can read more about noarr functions in the next section.


## Functions

A noarr function is a function that can be applied to a noarr [structure](Glossary.md#structure) using `|`.
The most important functions defined by the library are listed below.

### get_size

`get_size` returns the [size](Glossary.md#size) of the structure in bytes. Do not confuse this with length (described above).
A structure has one length per [dimension](Glossary.md#dimension); it is the number of valid indices.
On the other hand, a structure only has one size. It takes into accounts all parts of the structure, including the scalar type(s):

```cpp
std::size_t forty = my_structure_of_ten | noarr::get_size(); // 10 * sizeof(float)
std::size_t thirty_six = my_matrix_3x3 | noarr::get_size(); // 3 * 3 * sizeof(float)
```

### offset

`offset` retrieves offset of a scalar value. The result is in bytes again. You need to pass an [index](Glossary.md#index) in each [dimension](Glossary.md#dimension):

```cpp
std::size_t eight = my_structure_of_ten | noarr::offset<'i'>(2); // 2 * sizeof(float)
std::size_t twenty = my_matrix_3x3 | noarr::offset<'i', 'j'>(2, 1); // 5 * sizeof(float)
// 1 * 3 * sizeof(float) for the vector<'j'>, plus 2 * sizeof(float) for the vector<'i'>

// order of dimensions is not relevant
std::size_t also_twenty = my_matrix_3x3 | noarr::offset<'j', 'i'>(1, 2);
```

Alternatively, it is possible to pass all indices at once, using a [state](Glossary.md#state):

```cpp
auto state = noarr::idx<'i', 'j'>(2, 1);
std::size_t eight = my_structure_of_ten | noarr::offset(state); // this will ignore 'j', but OK
std::size_t twenty = my_matrix_3x3 | noarr::offset(state);
```

Later we will show examples when using state can be more useful.

### get_at

`get_at` is similar to `offset`, except that it returns an absolute reference instead of a relative offset.
Recall that a structure does not hold the data, it is just a description of the layout. As such, it needs a pointer to start with:

```cpp
// operator new is not noarr specific, it is a C++ construct, similar to std::malloc
void *data_ptr = operator new(my_structure_of_ten | noarr::get_size());
float &last_elem = my_structure_of_ten | noarr::get_at<'i'>(data_ptr, 9);
```

Similarly to `offset`, `get_at` can be used with a [state](Glossary.md#state):

```cpp
auto state = noarr::idx<'i'>(9);
float &last_elem = my_structure_of_ten | noarr::get_at(data_ptr, state);
```

### Other Functions

See [Other Functions](other/Functions.md) for the description of some non-essential functions, as well as a manual to defining your own.


## Bag

So far, we have mostly been working with the [structures](Glossary.md#structure) (ideas/layouts/types), but no actual memory (except for the unflattering `operator new` example).
Although these two aspects could always be held separately (as demonstrated above by [functions](#functions)), it is often more convenient to keep them together.
A bag is an object that remembers both. It is usually created using one of these two `make_bag` functions:

```cpp
auto unique_bag = noarr::make_bag(my_structure_of_ten);
auto ref_bag = noarr::make_bag(my_structure_of_ten, data_ptr);
```
The first variant allocates the memory (according to [`get_size`](#get_size)) and deallocates the memory in the bag's destructor.
The second variant uses a caller-supplied pointer. The caller must ensure there is enough space, and must deallocate the data after the bag is destroyed (the bag does neither).

### Indexing a Bag

A bag can be indexed using a [state](State.md) (similarly to [`get_at`](#get_at), except the bag already knows the data pointer):

```cpp
auto state = noarr::idx<'i'>(9);
float &last_elem = ref_bag[state]; // or unique_bag
```

It is still possible to pass the indices directly, but this time it gets slightly more complicated:

```cpp
float &last_elem = ref_bag.template at<'i'>(9); // or unique_bag
```

The `template` keyword may be omitted only when the type of the bag is known to the compiler.
This holds in simple cases where it is defined in the same function, but not for example here:

```cpp
template<typename Bag>
void foo(Bag bag) {
	float &last_elem = bag.template at<'i'>(9);
	// ...
}
```

### Copying or Moving a Bag

When defined as in the example above, `unique_bag` behaves like a `std::unique_ptr`. It cannot be copied, and moving it clears the original (makes it null).
On the other hand, `ref_ptr` behaves like a reference or raw pointer.
There is no difference between copying and moving (both the original and the new bag are valid and point to the same data, neither being the owner).

It is possible to get a reference bag pointing to a unique bag:

```cpp
auto ref_bag = unique_bag.get_ref(); // can be used on any bag type
```

In this case, `unique_bag` remains valid and retains the ownership. `ref_bag` is only valid as long as `unique_bag` (or more precisely the data it points to).

### Other uses

- The two components of a bag can be retrieved using `bag.structure()` and `bag.data()`, respectively.
- Any [function](#functions) can be applied to it (using the same `|` operator).
- Most [proto-structures](Glossary.md#proto-structure) too (using the same `^` operator), but there are some restrictions:
  - The proto-structure must not change the physical layout (e.g. [`vector`](structs/vector.md) is not allowed, but [`into_blocks`](structs/into_blocks.md) is).
  - The bag must be either a reference bag (see above) or a rvalue (e.g. a call to `make_bag`, `std::move`, or another `^`).


## Using algorithms with different structures

We would like to write algorithms in such a way that the structures they work with can be replaced.
For example, the following two structures can be used interchangeably:

```cpp
// i is row index, j is column index
auto matrix_columns = noarr::scalar<float>() ^ noarr::sized_vector<'i'>(nrows) ^ noarr::sized_vector<'j'>(ncols);
auto matrix_rows = noarr::scalar<float>() ^ noarr::sized_vector<'j'>(ncols) ^ noarr::sized_vector<'i'>(nrows);
```

Algorithms that can work with one should be able to work with the other. However, Noarr needs to know the structure at compile time (for performance).
So the right approach is to template all functions and then select between compiled versions. Here we make use of the fact that [dimensions](Glossary.md#dimension) are named:

```cpp
// function which does some logic templated by different structures
template<typename Structure>
void mul_by_scalar(Structure structure, void *data, float factor) {
	auto i_len = structure | noarr::get_length<'i'>();
	auto j_len = structure | noarr::get_length<'j'>();
	for(std::size_t i = 0; i < i_len; i++) {
		for(std::size_t j = 0; j < j_len; j++) {
			structure | noarr::get_at<'i', 'j'>(data, i, j) *= factor;
		}
	}
}
```

The same written with a bag:

```cpp
// function which does some logic templated by different structures
template<typename Bag>
void mul_by_scalar(Bag &bag, float factor) {
	auto i_len = bag | noarr::get_length<'i'>();
	auto j_len = bag | noarr::get_length<'j'>();
	for(std::size_t i = 0; i < i_len; i++) {
		for(std::size_t j = 0; j < j_len; j++) {
			bag[noarr::idx<'i', 'j'>(i, j)] *= factor;
		}
	}
}
```

The same written with a [traverser](Traverser.md) (which can select the order of the loops to work efficiently with both structures):

```cpp
// function which does some logic templated by different structures
template<typename Bag>
void mul_by_scalar(Bag &bag, float factor) {
	noarr::traverser(bag).for_each([&](auto state) {
		bag[state] *= factor;
	});
}
```

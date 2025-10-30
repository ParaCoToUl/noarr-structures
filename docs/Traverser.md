# Traverser

Traverser is a tool for multi-dimensional [structure](Glossary.md#structure) iteration.
It can process individual elements ([`for_each(lambda)`](#for_eachlambda)) or whole sections ([`for_dims<...>(lambda)`](#for_dimslambda)) of a structure.
It can also work with over [multiple structures](#traversing-multiple-structures-at-once) at once.
Traverser is also the library's main tool for data parallelism.


## `for_each(lambda)`

In the following example, we use a simple call to `for_each` to iterate over a 2D structure.

```cpp
// Usual matrix allocation
auto matrix_struct = noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>();
auto matrix = noarr::make_bag(matrix_struct);

// Traverser example: zero out the matrix
noarr::traverser(matrix).for_each([&](auto s) {
	// This code executes for each element, matrix[s]
	matrix[s] = 0;
});
```

The lambda function is called repeatedly, once for each element, in a well-defined order, described [below](#traversing-order).
The lambda parameter `s` is a [state](State.md) containing the indices for `'i'` and `'j'`.
Note that indexing a [bag](BasicUsage.md#bag) is by far not the only way to use a state, see [state usages summary](State.md#usages).


## `for_dims<...>(lambda)`

An alternative to `for_each` is `for_dims`, which also allows you to iterate over rows, columns, etc, instead of individual elements.
The following example counts the columns in a really impractical way:

```cpp
std::size_t num_cols = 0;
noarr::traverser(matrix).template for_dims<'j'>([&](auto t) {
	// This code executes for each column, that is, for each distinct j
	num_cols++;
});
assert(num_cols == (matrix | noarr::get_length<'j'>()));
```

Unlike `for_each`, the lambda in `for_dims` receives another `traverser` as a parameter (`t` in this example).
This inner traverser comes with the `'j'` index already fixed (to a different value in each outer traverser iteration).
This makes it easy to do a nested call to either `for_each` or another `for_dims`.
The inner traverser will only iterate the dimensions that are not fixed.

In the following example, we nest a `for_each` call in a `for_dims` call.
The outer `for_dims` call iterates all `'j'` while the inner `for_each` calls iterate all `'i'`.

```cpp
// Normalize each column separately
noarr::traverser(matrix).template for_dims<'j'>([&](auto t) {
	// This code executes for each column:
	// t is a traverser with 'j' fixed

	double norm2 = 0;
	t.for_each([&](auto s) {
		// This code executes for each element in that column:
		// s is a state with both 'i' and 'j'
		double elem = matrix[s];
		norm2 += elem * elem;
	});
	double norm = std::sqrt(norm2);

	t.for_each([&](auto s) {
		matrix[s] /= norm;
	});
});
```

It is possible to specify multiple dimensions in `for_dims`.
In most cases, it can be any subset of the structure dimensions, in any order (but see [Traversing order](#traversing-order)).
The following example sketches some ways to traverse a 3D array:

```cpp
auto a3d = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>() ^ noarr::array<'k', 500>());

// Iterate all possible indices for 'j'
noarr::traverser(a3d).template for_dims<'j'>([&](auto trav) {
	// trav has 'j' fixed (what remains is 'k' and 'i')

	// Iterate all possible indices for 'k' and 'i'
	trav.for_each([&](auto state) {
		// state has 'i', 'j', 'k'
	});
});

// Iterate all possible indices for 'k' and 'i'
noarr::traverser(a3d).template for_dims<'k', 'i'>([&](auto trav) {
	// trav has 'k' and 'i' fixed (what remains is 'j')

	// Iterate all possible indices for 'j'
	trav.for_each([&](auto state) {
		// state has 'i', 'j', 'k'
	});
});

// Iterate all possible indices for 'i'
noarr::traverser(a3d).template for_dims<'i'>([&](auto trav) {
	// trav has 'i' fixed (what remains is 'k' and 'j')

	// Iterate all possible indices for 'j'
	trav.template for_dims<'j'>([&](auto inner_trav) {
		// inner_trav has both 'i' and 'j' fixed (what remains is 'k')

		// Iterate all possible indices for 'k'
		inner_trav.for_each([&](auto state) {
			// state has 'i', 'j', 'k'
		});
	});
});
```

The following edge cases are possible, too:

```cpp
// Iterate all possible indices for 'i', 'j', 'k', effectively iterating all elements
noarr::traverser(a3d).template for_dims<'i', 'j', 'k'>([&](auto trav) {
	// trav has 'i', 'j', 'k' fixed (nothing remains)

	// This traversal will have exactly one iteration (it will not add any dimensions)
	trav.for_each([&](auto state) {
		// state has 'i', 'j', 'k'
	});
});

// Perform one iteration, passing an equivalent traverser to the lambda
noarr::traverser(a3d).template for_dims<>([&](auto trav) {
	// trav has no dimensions fixed, it is the same as the outer `noarr::traverser(matrix)`

	// Iterate all possible indices for 'i', 'j', 'k'
	trav.for_each([&](auto state) {
		// state has 'i', 'j', 'k'
	});
});
```


## `state()`: obtaining a plain state in `for_dims`

The `state()` member function can be called on a traverser to obtain its fixed indices.
This allows you to use `for_dims` similarly to `for_each`. Reusing the first example:

```cpp
auto matrix_struct = noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>();
auto matrix = noarr::make_bag(matrix_struct);

noarr::traverser(matrix).for_each([&](auto s) {
	matrix[s] = 0;
});

noarr::traverser(matrix).template for_dims<'j', 'i'>([&](auto t) {
	// We can use traverser as if it was a state
	matrix[t] = 0;

	// Or we can extract the state explicitly
	auto s = t.state();
	matrix[s] = 0;
});
```

Note that `state()` can be called on any traverser, but it may not always be directly usable to index the structure.
For example:

```cpp
noarr::traverser(matrix).template for_dims<'i'>([&](auto t) {
	auto s = t.state(); // This is OK, s contains just 'i'
	matrix[s] = 0; // This fails, s does not contain 'j'!
});
```

In particular, a freshly created traverser will always return an empty state.

There are other [usages of state](State.md#usages), many of which do not require a "complete" state.


## `order(proto-structure)`: customizing the traversal

The `order(...)` member function applies a [proto-structure](Glossary.md#proto-structure) to the current structure, thus changing how it will be traversed.
The function returns a new traverser, the current traverser is *not* modified.

The following example shows two possible ways to split an array [into blocks](structs/into_blocks.md),
one using `order` and the other wrapping the structure manually.
Note that in this simple case, splitting into blocks will not have any significant effect. It is done just for demonstration purposes.

```cpp
auto original = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>());

// Create a proto-structure that splits dimension 'i' into blocks of size 4,
//   the resulting structure will have dimensions 'y' (index of block) and 'x' (index of element within block)
auto blk = noarr::into_blocks<'i', 'y', 'x'>(4);

// Variant 1:
noarr::traverser(original).order(blk).for_each([&](auto state) {
	// state contains 'i' and can therefore be used with the original structure
	original[state] = 0;

	// state does not contain 'x' or 'y'
	//   (these were converted into 'i' internally by traverser)
});

// Variant 2:
auto blocked = original.get_ref() ^ blk;

noarr::traverser(blocked).for_each([&](auto state) {
	// state contains 'x' and 'y' so it must be used with the corresponding manually wrapped structure
	blocked[state] = 0;

	// state does not contain 'i', because the traverser cannot know about 'i'
	//   (all it got was a structure with dimensions 'x' and 'y')
	/* This fails during compilation: */ original[state] = 0;
});
```

The difference between the two variants may seem subtle, but it is important. The traverser always returns states relative to the initial structures,
this is the reason why in the second variant, the state cannot be used to index `original`; the traverser was constructed from `blocked`, which contains
just `'x'` and `'y'`, while `original` expects a state containing `'i'`.

Whether `order` should be used (instead of wrapping the structure manually) depends on the situation.
In some applications, it may be necessary to extract indices from the state for some other use apart from structure indexing.
In these cases, one of the options may be conceptually much more appropriate than the other.

The `order` function can be applied multiple times, `.order(A).order(B)` is equivalent to `.order(A ^ B)`.
The dimension names passed to `for_dims` are looked up in the updated structure.
For example, in the above snippet, `for_dims<'x', 'y'>` would be appropriate in both variants, while `for_dims<'i'>` in neither.
The inner traverser's `.state()` is the same as the state passed to `for_each`.
In the above snippet, it would have `'i'` in the first variant, and `'x'`, `'y'` in the second variant.

### Usages of `order`

As mentioned above, [`noarr::into_blocks`](structs/into_blocks.md) does not do much on its own.
In combination with `for_dims`, it can be used to perform some action for each block or to change the traversing order in multidimensional structures.
It can also be composed with other structures to generate more observable effects or performance gains.

[`noarr::hoist`](structs/hoist.md) and [`noarr::merge_zcurve`](structs/merge_zcurve.md) can be used to change the traversing order.

[`noarr::fix`](structs/fix.md), [`noarr::shift`](structs/shift.md), [`noarr::slice`](structs/slice.md), [`noarr::step`](structs/step.md),
and their variants can be used to limit the traversal to a subset of elements.

Almost any proto-structure can be used, although with varying usefulness.
Proto-structures that change the in-memory layout (e.g. [`noarr::array`](structs/array.md), which repeats the layout) are forbidden in `order` and won't compile.
Others are not useful at all -- for example [`noarr::rename`](structs/rename.md) (the renaming would only matter in `for_dims` template arguments).

[`noarr::bcast`](structs/bcast.md) could be used to traverse the structure repeatedly, but the index of the iteration would not be available in the state.
In this case, you can use the manual variant, `noarr::traverser(orig_struct ^ noarr::bcast<'n'>(...))`,
and still use just `orig_struct` in the lambda body, since `orig_struct` will ignore `'n'` anyway.

Note that, if the dimensions added by the applied proto-structures are to be used in `for_dims` (or `for_each`),
it is always necessary to apply the proto-structure to the original structure instead of using `order` on the traverser.

See the linked documentation pages for usage examples and detailed descriptions.


## Traversing order

A call to `for_each` or `for_dims` is equivalent to a nested loop where each layer corresponds to one dimension.
In `for_dims`, the number of layers and their order is the same as the number and order of dimension names in the template argument.

In `for_each`, the order depends on the structure (or more precisely its [signature](Signature.md)). The outermost dimensions are iterated first.
In most structures, this results in the elements being traversed in-order as they are stored in memory.
[`noarr::hoist`](structs/hoist.md) can be used to change the signature to an equivalent one with different order of dimensions.

Tuple-like dimensions (see [Dimension Kinds](DimensionKinds.md)) are iterated by a "static loop" -- a loop is expanded using template specialization.
It is not allowed to move tuple-like dimensions down (i.e. pull tuple elements up) when specifying the order in `for_dims` or [`noarr::hoist`](structs/hoist.md).


## Traversing multiple structures at once

The constructor can take any number of structures. The structures can share some dimension names but not all structures need to have all dimensions.
When multiple structures share a dimension, they must also have the same length in that dimension, otherwise the behavior is undefined.
Each dimension is iterated only once and the state will only contain one index for that dimension, regardless of how many structures it appears in.

```cpp
// Structure copy, all dimensions are shared
auto from = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>(), from_data);
auto to = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'j', 400>() ^ noarr::array<'i', 300>());

noarr::traverser(from, to).for_each([&](auto s) {
	to[s] = from[s];
});

// Matrix multiplication, each dimension is shared by just two structures
auto a = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>(), a_data);
auto b = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'j', 400>() ^ noarr::array<'k', 500>(), b_data);
auto c = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'k', 500>());

noarr::traverser(c).for_each([&](auto s) {
	c[s] = 0;
});
noarr::traverser(a, b, c).for_each([&](auto s) {
	c[s] += a[s] * b[s];
});
```


## Using bare structures without bags

Traverser does not need a bag. All necessary information is already present in the structure.
The first example in this document could be written equivalently as:

```cpp
auto matrix_struct = noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>();
auto matrix = noarr::make_bag(matrix_struct);

// Note: here we passed matrix (not matrix_struct) in the first example
noarr::traverser(matrix_struct).for_each([&](auto s) {
	// We still use the bag on this line
	matrix[s] = 0;
});
```

A bag is just a structure and a pointer and the [`operator[]`](BasicUsage.md#indexing-a-bag) on bag is just a shortcut for a [`get_at`](BasicUsage.md#get_at) call:

```cpp
auto matrix_struct = noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>();
void *matrix_ptr = operator new(matrix_struct | noarr::get_size());

noarr::traverser(matrix_struct).for_each([&](auto s) {
	float &elem = matrix_struct | noarr::get_at(matrix_ptr, s);
	elem = 0;
});
```


## Traverser iterator

In addition to the `for_*` methods, a traverser can also be used in a [range-based for loop](https://en.cppreference.com/w/cpp/language/range-for).
For this to work, it is necessary to include `<noarr/structures/interop/traverser_iter.hpp>`.
The loop will only use the top-most dimension and it provides an inner traverser in each iteration (similarly to `for_dims`).
For example:

```cpp
auto matrix = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>());

for(auto trav : noarr::traverser(matrix)) {
	// This code executes for each column:
	// t is a traverser with 'j' fixed

	trav.for_each([&](auto state) {
		// Only now we have the index in i
		matrix[state] = 0;
	});

	// ~or~

	for(auto unit_trav : trav) { // Iterate the topmost dimension of trav, leaving no dimensions
		matrix[unit_trav] = 0;
	}
}
```

To iterate over each element using just for loops, you would need to nest as many loops as there are dimensions in the structure.
It is not necessary to come up with a new name every time. The following example might be packed too much, it is here just to show what is possible:

```cpp
auto cube = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 25>() ^ noarr::array<'j', 30>() ^ noarr::array<'k', 35>());

for(auto trav : noarr::traverser(cube)) for(auto trav : trav) for(auto trav : trav) {
	cube[trav] = 0;
}
```

### Use with OpenMP

For loops can be used with [OpenMP](https://www.openmp.org/).
The first traverser iterator example [above](#traverser-iterator) can be modified by simply adding an OMP pragma:

```cpp
auto matrix = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>());

#pragma omp parallel for
for(auto trav : noarr::traverser(matrix)) {
	// This code executes for each column:
	// t is a traverser with 'j' fixed

	trav.for_each([&](auto state) {
		// Only now we have the index in i
		matrix[state] = 0;
	});

	// ~or~

	for(auto unit_trav : trav) { // Iterate the topmost dimension of trav, leaving no dimensions
		matrix[unit_trav] = 0;
	}
}
```

If you want to avoid the extra loop and indentation, you can use the following template:

```cpp
template<typename T, typename F>
inline void omp_trav_for_each(const T &trav, const F &f) {
	#pragma omp parallel for
	for(auto trav_inner : trav)
		trav_inner.for_each(f);
}
```

Then just replace serial code like `t.for_each([](auto state) { ... })` with `omp_trav_for_each(t, [](auto state) { ... })` (where `t` is the traverser).


## Traverser range and TBB integration

The range of the topmost dimension used in the previous section can also be extracted explicitly using the `.range()` method:

```cpp
auto matrix = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 300>() ^ noarr::array<'j', 400>());

for(auto trav : noarr::traverser(matrix).range()) {
	/* ... */
}
```

Traverser range supports additional features not available in traverser directly:

- An optional template argument can be added to the `.range()` call to explicitly select a dimension, e.g. `.template range<'i'>()`.
- The range object has `.begin_idx` and `.end_idx` fields that can be used to limit the iteration range.
  They are initialized to `0` and the length, respectively, so by default, no elements are omitted.
- The range object can be converted to a traverser using `as_traverser()`. The new traverser will honor `begin_idx` and `end_idx`.
  This is equivalent to calling `.order` with a [`noarr::slice`](structs/slice.md) on the original traverser.
- The range object satisfies some (not all) requirements for a [container](https://en.cppreference.com/w/cpp/named_req/Container) (`size`, `empty`, member types),
  and can be indexed using `[]` (returning inner traversers as in iteration).

When also including `<noarr/structures/interop/tbb.hpp>`, the range can be used for parallelization with [Threading Building Blocks](https://intel.com/oneTBB):

```cpp
tbb::parallel_for(noarr::traverser(matrix).range(), [&](auto subrange) {
	for(auto trav : subrange) {
		// Note: the original structure was two-dimensional,
		//   we have only iterated one dimension - the other one still remains
		for(auto trav : trav) {
			matrix[trav] = 0;
		}
	}

	// ~or~

	// Convert to traverser, and continue as usual
	subrange.as_traverser().for_each([&](auto state) {
		matrix[state] = 0;
	});
});
```

If the only thing you would do with the traverser is a `for_each`, there is also a shortcut available:

```cpp
// Note: we don't pass a range here, just the traverser
noarr::tbb_for_each(noarr::traverser(matrix), [&](auto state) {
	matrix[state] = 0;
});
```

### Parallel reduction

Noarr provides a function `noarr::tbb_reduce` for parallel with non-scalar result (i.e. reducing to a structure, not just one value).
In the following example, we use it to sum the rows and columns of a matrix:

```cpp
auto matrix = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'j', 300>() ^ noarr::array<'i', 400>(), matrix_data);
auto row_sums = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 400>()); // A column vector
auto col_sums = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'j', 300>()); // A row vector

// Row sums - serial version for comparison:

noarr::traverser(row_sums).for_each([&](auto state) {
	row_sums[state] = 0;
});

noarr::traverser(matrix).for_each([&](auto state) {
	row_sums[state] += matrix[state];
});

// Row sums - parallel version:

noarr::tbb_reduce(
	// Input traverser
	noarr::traverser(matrix),

	// Fill the output with neutral elements (zeros) with respect to the operation (+)
	[](auto state, auto &out) {
		out[state] = 0; // out is a bag: row_sums or its copy
	},

	// Accumulate elements into (partial) result
	[matrix](auto state, auto &out) {
		out[state] += matrix[state];
	},

	// In-place join partial results
	[](auto state, auto &out_left, const auto &out_right) {
		out_left[state] += out_right[state];
	},

	// Output bag
	row_sums
);

// Column sums - parallel version:

noarr::tbb_reduce(/* ... all args the same ... */, col_sums);
```

These functions automatically detect the parallelization strategy.
The input structure is always split according to its outermost dimension, in this case `'i'` (in both reductions).

![When parallelizing, accesses to `row_sums` go to different elements, while accesses to `col_sums` go to overlapping sets of elements](img/trav-tbb-reduce.svg)

Then, depending on the output structure, the library decides whether it is necessary to create multiple thread-local copies of the output structure.
In the above example, the reduction into `row_sums` can be done on one copy without a conflict, since two threads never use the same `'i'`,
and therefore they never access the same element of `row_sums`.
On the other hand, `col_sums` will be copied (one copy per thread), because the structure only has the `'j'` dimension (it ignores the index in `'i'`).
Although two threads never use the same `'i'`, they can (and often will) use the same `'j'` at the same time, therefore accessing the same element.

The two structures need not be related and the output structure may be accessed using a different set of dimensions.
In the following example, the input structure only has an `'i'` dimension, while the output structure only has `'v'`:

```cpp
// Vector of values in range [0, 256), indexed by some 'i'
auto values = noarr::make_bag(noarr::scalar<std::uint8_t>() ^ noarr::vector<'i'>(size), values_data);
// Histogram of the values, indexed by the value ('v'), gives the number of occurrences
auto histogram = noarr::make_bag(noarr::scalar<std::size_t>() ^ noarr::array<'v', 256>());

noarr::tbb_reduce(
	// Input traverser
	noarr::traverser(values),

	// Fill the output with neutral elements
	[](auto histo_state, auto &histo) {
		histo[histo_state] = 0;
	},

	// Accumulate elements into (partial) result
	[values](auto values_state, auto &histo) {
		std::uint8_t value = values[values_state];
		histo[noarr::idx<'v'>(value)] += 1; // Note: cannot use values_state for histo
	},

	// In-place join partial results
	[](auto histo_state, auto &histo_left, const auto &histo_right) {
		histo_left[histo_state] += histo_right[histo_state];
	},

	// Output bag
	histogram
);
```


## CUDA integration

The `noarr::cuda_threads` template can be used to bind some dimension names in a traverser to CUDA threads/blocks.

The function is called in host code, with template parameters specifying the dimension names, and one function parameter, which is a traverser.
There can be 2, 4, or 6 dimension names, which are respectively bound to these CUDA coordinates: block X, thread X, block Y, thread Y, block Z, thread Z.

The function returns a new traverser (`inner()`), together with kernel launch parameters (`grid_dim()` and `block_dim()`).
The new traverser must only be used in device code (e.g. the kernel function).

In the following simplified example of matrix multiplication, we bind four dimension names to the input matrices' dimensions:

```cu
template<typename T, typename A, typename B, typename C>
__global__ void matmul(T trav, A a, B b, C c) {
	float result = 0;

	trav.for_each([=, &result](auto state) { // For each j
		result += a[state] * b[state];
	});

	c[trav] = result;
}

// a_data, b_data, c_data are pointers to memory accessible from the GPU device
auto a = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 3000>() ^ noarr::array<'j', 4000>(), a_data);
auto b = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'j', 4000>() ^ noarr::array<'k', 5000>(), b_data);
auto c = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 3000>() ^ noarr::array<'k', 5000>(), c_data);

auto block_size = noarr::lit<8>;
auto blk_order = noarr::into_blocks<'i', 'I', 'i'>(block_size) ^ noarr::into_blocks<'k', 'K', 'k'>(block_size);

auto cutrav = noarr::cuda_threads<'I', 'i', 'K', 'k'>(noarr::traverser(a, b, c).order(blk_order));

matmul<<<cutrav.grid_dim(), cutrav.block_dim()>>>(cutrav.inner(), a, b, c.get_ref());
```

As noted in [`noarr::into_blocks` documentation](structs/into_blocks.md), this structure does not work well if the length is not a multiple of block size.
If that condition is not satisfied, one can use [`noarr::into_blocks_dynamic`](structs/into_blocks.md#into_blocks_dynamic).
Each `into_blocks_dynamic` creates one more dimension (`'r'` and `'s'` below). The added dimensions must be checked inside of the kernel:

```cu
template<typename T, typename A, typename B, typename C>
__global__ void matmul(T trav, A a, B b, C c) {
	trav.template for_dims<'r', 's'>([=](auto trav) {
		float result = 0;

		trav.for_each([=, &result](auto state) { // For each j
			result += a[state] * b[state];
		});

		c[trav] = result;
	});
}

// a_data, b_data, c_data are pointers to memory accessible from the GPU device
auto a = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 3500>() ^ noarr::array<'j', 4500>(), a_data);
auto b = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'j', 4500>() ^ noarr::array<'k', 5500>(), b_data);
auto c = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 3500>() ^ noarr::array<'k', 5500>(), c_data);

auto block_size = noarr::lit<8>;
auto blk_order = noarr::into_blocks_dynamic<'i', 'I', 'i', 'r'>(block_size) ^ noarr::into_blocks_dynamic<'k', 'K', 'k', 's'>(block_size);

auto cutrav = noarr::cuda_threads<'I', 'i', 'K', 'k'>(noarr::traverser(a, b, c).order(blk_order));

matmul<<<cutrav.grid_dim(), cutrav.block_dim()>>>(cutrav.inner(), a, b, c.get_ref());
```

# step

Select a subsequence of [indices](../Glossary.md#index) in the specified [dimension](../Glossary.md#dimension) using a specified starting index and step (gap length).

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, typename T, typename StartT, typename StepT>
struct noarr::step_t;

constexpr proto noarr::step(auto start, auto step);

template<char Dim>
constexpr proto noarr::step(auto start, auto step);
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

A `step_t` structure is similar to its wrapped `T` structure, except that it may be shorter in dimension `Dim`.
Some elements elements are omitted from the resulting view. Namely, only the elements with the index (*a*·*k* + *b*) in `Dim` are preserved,
where *a* and *b* are the parameters of the structure, and *k* is a nonnegative integer. *k* becomes the new index in `Dim` for the element.

The length of the new structure in `Dim` depends on both parameters and the original length.
It is the first *k* such that (*a*·*k* + *b*) is not a valid index in the original dimension.
This preserves the usual property that valid indices are exactly those greater than or equal to zero and (strictly) less than the length.

The `step` function creates a proto-structure for `step_t`.
The dimension name, `Dim` is optional for this function, it defaults to the outer-most dimension of the original structure.

Both parameters are unsigned integers. `start` must be (strictly) less than `step` (thus `step` must always be nonzero).

See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `start` and `step` (and `StartT` and `StepT` respectively).


## Usage examples

Creating `N` structures using `noarr::step<Dim>(i, N)` (each with a different `i` from 0 inclusive to `N` exclusive) effectively splits the original structure
into `N` parts that are disjunct (share no elements) and together cover all the elements of the original structure.
Additionally, if the original length in `Dim` is large enough in comparison to `N`, the resulting parts will have roughly the same number of elements.

This can be used to split work among threads or processing units. [`noarr::cuda_step`](cuda_step.md) is an application of `noarr::step` for CUDA kernels.

In the following example, we split `all` into 4 disjoint parts covering the original structure:

```cpp
auto all = noarr::scalar<float>() ^ noarr::vector<'i'>(42);

auto part0 = all ^ noarr::step(0, 4); // 0,  4,  8, ..., 36, 40
auto part1 = all ^ noarr::step(1, 4); // 1,  5,  9, ..., 37, 41
auto part2 = all ^ noarr::step(2, 4); // 2,  6, 10, ..., 38
auto part3 = all ^ noarr::step(3, 4); // 3,  7, 11, ..., 39

assert((part0 | noarr::get_length<'i'>()) == 11);
assert((part1 | noarr::get_length<'i'>()) == 11);
assert((part2 | noarr::get_length<'i'>()) == 10);
assert((part3 | noarr::get_length<'i'>()) == 10);

assert((part3 | noarr::offset<'i'>(7)) == (all | noarr::offset<'i'>(3 + 4*7)));
```

`noarr::step` is equivalent to `noarr::step<'i'>`, since it is the only dimension.

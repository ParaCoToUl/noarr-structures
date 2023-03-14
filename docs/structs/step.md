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

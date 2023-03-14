# shift

Shift the [index](../Glossary.md#index) in the specified [dimension](../Glossary.md#dimension) by some fixed nonnegative number (thus skipping some leading elements).

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, typename T, typename DeltaT>
struct noarr::shift_t;

template<char... Dims>
constexpr proto noarr::shift(auto... deltas);
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

A `shift_t` structure is similar to its wrapped `T` structure, except that it may be shorter in dimension `Dim`.
Some leading elements are omitted from the resulting view. The number of omitted elements depends on the specified delta.
The length of the new structure in `Dim` is shortened by the delta (the starting point).
The new [indices](../Glossary.md#index) run from 0 (inclusive) to the new length (exclusive), as always.
Each element will obtain an index shifted by the constant offset: new index = old index − delta.

Note that the memory layout is not modified -- only the view is changed.

The `shift` function can accept a list of dimensions: it will compose multiple `shift_t`s if necessary.

Shifting a dimension is similar to [slicing](slice.md) it, except that only the starting element is shifted, while the last element stays the same.

It is possible to shift by 0 (yielding a structure equivalent to the original), or by the full length (yielding a structure with no accessible elements).
The delta is an unsigned integer: it is not possible to shift by a negative number.
Shifting with number large enough and indexing with indices large enough to cause wraparound is undefined by noarr and can in some cases result in undefined behavior.

See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `deltas` (and `DeltaT`).

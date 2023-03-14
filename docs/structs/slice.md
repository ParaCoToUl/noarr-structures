# slice

Slice the structure in the specified [dimension](../Glossary.md#dimension) (thus skipping some leading and some trailing elements).

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, typename T, typename StartT, typename LenT>
struct noarr::slice_t;

template<char Dim>
constexpr proto noarr::slice(auto start, auto length);
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

A `slice_t` structure is similar to its wrapped `T` structure, except that it may be shorter in dimension `Dim`.
Some leading elements and some trailing elements are omitted from the resulting view. The number of omitted elements depends on the specified delta.
The length of the new structure in `Dim` is specified by the second parameter.
The new [indices](../Glossary.md#index) run from 0 (inclusive) to the new length (exclusive), as always.
Each element will obtain an index shifted by the constant offset: new index = old index − starting point.

*The second parameter is the length, not the end index.* This is to allow the length to be constexpr (see below) even if the starting point is not.

Note that the memory layout is not modified -- only the view is changed.

The `slice` function creates a proto-structure for `slice_t`.

It is possible, for example, to slice out an empty range (`length = 0`) or the whole structure (`start = 0`, `length = *the original length*`).
However, `start` must be an unsigned number and `start + length` must be less than or equal to the original size (using unsigned wraparound is not allowed here).

If you need a single-ended slice (i.e. change the beginning but leave the end unchanged), use [`noarr::shift`](shift.md).

See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `start` and `length` (and `StartT` and `LenT` respectively).

# hoist

Move the specified [dimension](../Glossary.md#dimension) to the top level (for [multidimensional iteration](../Traverser.md)).

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, typename T>
struct noarr::hoist_t;

template<char Dim>
constexpr proto noarr::hoist();
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

A structure of type `hoist_t` has the same properties as its sub-structure `T`, except that `Dim` appears on the top.
The layout is the same and offsets of individual elements are computed in the same way.
The only difference between `T` and `hoist_t<Dim, T>` is how it appears from outside,
which has an effect on the preferred [traversal](../Traverser.md) order.
In case of single-threaded traversal, the hoisted dimension (`Dim`) will be iterated in the outer-most loop.
In case of parallelization, the hoisted dimension will be the one according to which the structure is split.

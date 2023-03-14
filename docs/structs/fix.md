# fix

Set the [index](../Glossary.md#index) in the specified [dimension](../Glossary.md#dimension) to some fixed number.

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, typename T, typename IdxT>
struct noarr::fix_t;

template<char... Dims>
constexpr proto noarr::fix(auto... indices);

constexpr proto noarr::fix(auto state);
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

The `fix_t` structure removes dimension `Dim` from structure `T`, replacing its index with a fixed value of type `IdxT`.
The resulting view has a smaller dimensionality: for a matrix, it views just a row/column, for a 3D matrix it views just one 2D layer of it, etc.

The `fix` function can accept a list of dimensions: it will compose multiple `fix_t`s if necessary.
The second overload accepts a [state](../State.md) instead. It fixes all indices available in the state.
See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `indices` (and `IdxT`).

Note that it is not necessary to use `fix` to pass an index to a structure:
passing it in [state](../State.md) or directly to a [function](../BasicUsage.md#functions) is often sufficient.

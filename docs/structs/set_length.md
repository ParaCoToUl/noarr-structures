# set_length

Set the [length](../Glossary.md#length) in the specified [dimension](../Glossary.md#dimension) to some fixed number.

```hpp
#include <noarr/structures_extended.hpp>

template<auto Dim, typename T, typename LenT>
struct noarr::set_length_t;

template<auto... Dims>
constexpr proto noarr::set_length(auto... lengths);
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

The `set_length_t` structure is the same as structure `T` except that the length in dimension `Dim` is set.
The length must be unknown for `T` -- setting an already set length results in a compile-time error.

The `set_length` function can accept a list of dimensions: it will compose multiple `set_length_t`s if necessary.

See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `lengths` (and `LenT`).

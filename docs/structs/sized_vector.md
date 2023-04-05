# sized_vector

Repeat the [sub-structure](../Glossary.md#sub-structure) layout, forming a vector of elements matching the sub-structure,
and use a new [dimension](../Glossary.md#dimension) (of the specified length) to select between the elements.

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim>
constexpr proto noarr::sized_vector(auto length);
// = noarr::vector<Dim>() ^ noarr::set_length<Dim>(length)
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

The `sized_vector` function is a shortcut for a composition of [`noarr::vector`](vector.md) and [`noarr::set_length`](set_length.md).
It is the easiest way to define a noarr structure. For more general tools, see the mentioned components.

See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `length`.
Also, in case the length is known at compile time, [`noarr::array`](array.md) can be used instead.

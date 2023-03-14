# array

Repeat the [sub-structure](../Glossary.md#sub-structure) layout, forming an array of elements matching the sub-structure,
and use a new [dimension](../Glossary.md#dimension) (of the specified constant length) to select between the elements.

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, std::size_t Len, typename T>
using noarr::array = /*...*/;

template<char Dim, std::size_t Len>
constexpr proto noarr::array();
// = noarr::sized_vector<Dim>(lit<Len>)
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

`array` is a shortcut for [`noarr::sized_vector`](sized_vector.md) of [static](../Glossary.md#static-value) [length](../Glossary.md#length).
Along with `sized_vector`, it is the easiest way to define a noarr structure.
For more general tools (and more information about the underlying structures),
see [`noarr::vector`](vector.md) and [Lengths section in Basic Usage](../BasicUsage.md#lengths).

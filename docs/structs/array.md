# array

Repeat the [sub-structure](../Glossary.md#sub-structure) layout, forming an array of elements matching the sub-structure,
and use a new [dimension](../Glossary.md#dimension) (of the specified constant length) to select between the elements.

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, std::size_t Len, typename T>
using noarr::array = /*...*/;

template<char Dim, std::size_t Len>
constexpr proto noarr::array();
// = noarr::vector<Dim>(lit<Len>)
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

`array` is a shortcut for [`noarr::vector`](vector.md) of [static](../Glossary.md#static-value) [length](../Glossary.md#length).
Along with `vector`, it is the easiest way to define a noarr structure.
For more general tools (and more information about the underlying structures),
see [`noarr::vector`](vector.md) and [Lengths section in Basic Usage](../BasicUsage.md#lengths).


## Usage examples

Use array for homogeneous "tuples", i.e. structures with a constant size of the same type/structure:

```cpp
// the coordinates of a point in 3D (x, y, z)
auto point = noarr::scalar<float>() ^ noarr::array<'i', 3>();
// color, dim 'c' is the channel (for r, g, b, alpha)
auto color = noarr::scalar<std::uint8_t>() ^ noarr::array<'c', 4>();
// a complex number (when represented using a structure)
auto complex = noarr::scalar<float>() ^ noarr::array<'c', 2>();
```

In these cases, it would also be possible to use a plain scalar where the element type is a C++ structure.
The `complex` example is perhaps the most distinct instance (`noarr::scalar<std::complex>()`).
However, when we use `noarr::array`, we can further customize the layout:

```cpp
auto complex_proto = noarr::array<'c', 2>();

// a complex vector, array-of-structures (the usual representation)
auto complex_vec_aos = noarr::scalar<float>() ^ complex_proto ^ noarr::vector<'i'>();
// structure-of-arrays (an alternative representation)
auto complex_vec_soa = noarr::scalar<float>() ^ noarr::vector<'i'>() ^ complex_proto;
```

Both `complex_vec_aos` and `complex_vec_soa` can be used in the same way.

Another use for `noarr::array` is a structure whose length is not chosen for its significance to the algorithm,
but rather as a constant tuned so as to make the algorithm perform the best.
See [the vectorization example in `noarr::into_blocks`](into_blocks.md#parallelization).

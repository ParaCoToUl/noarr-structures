# bcast

Add an ignored [dimension](../Glossary.md#dimension) to a structure.

```hpp
#include <noarr/structures_extended.hpp>

template<auto Dim, typename T>
struct noarr::bcast_t;

template<auto... Dims>
constexpr proto noarr::bcast();

template<auto... Dims>
constexpr proto noarr::bcast(auto... lengths);
// = noarr::bcast<Dims...>() ^ noarr::set_length<Dims...>(lengths)
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

The `bcast_t` structure adds to structure `T` dimension `Dim`, which is not involved in any size or offset calculations.
Effectively, this makes the structure appear as if it were repeated once for each index in the new dimension.
Each one element of the original structure will be broadcast to multiple elements of the new structure
(the coordinates of these new elements will only differ in the newly added dimension `Dim`).

Note that the memory layout is not modified and (in case of bags) no data are copied -- only the view is changed.

The `bcast` function can accept a list of dimensions: it will compose multiple `bcast_t`s if necessary.

Neither `bcast_t` itself nor the first overload of `bcast` set the length of the new dimension - it must be [set externally](../BasicUsage.md#lengths).
The second overload of `bcast` provides a shortcut for this by setting the length in the structure immediately using [`noarr::set_length`](set_length.md).
See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `lengths`.


## Usage examples

This structure can be used to have a [traverser](../Traverser.md) visit each element repeatedly:

```cpp
auto structure = noarr::scalar<float>() ^ noarr::vector<'i'>(42);

noarr::traverser(structure ^ noarr::bcast<'r'>(5)).for_each([&](auto state) {
	int round = noarr::get_index<'r'>(state);

	// Fine, structure will ignore 'r' (same with e.g. bag[] or get_at)
	std::size_t off = structure | noarr::offset(state);
});
```

Broadcast can also be used to make a structure appear as having more dimensions without really changing the dimensionality of the data.
Specifically, if the length is set to one (`noarr::bcast<'...'>(1)`), it will still appear as having the same elements.

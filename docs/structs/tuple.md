# tuple

Concatenate multiple structure layouts and use a new [dimension](../Glossary.md#dimension)
to select between the original structures (now elements of the whole).

```hpp
#include <noarr/structures.hpp>

template<auto Dim, typename... Ts>
struct noarr::tuple;

template<auto Dim>
constexpr noarr::tuple<Dim, /*...*/> make_tuple(auto... ts);
```


## Description

The layout of a `tuple` structure consists of the layouts of each of `Ts`, in order, one after another without any padding.
The whole structure has one additional dimension, named `Dim`. The [index](../Glossary.md#index) in `Dim` is used to select one of the `Ts`.

In order to use any of the [sub-structures](../Glossary.md#sub-structure) (`Ts`),
it is necessary to [set the index](../BasicUsage.md) in `Dim`, as otherwise, some operations would be ambiguous.
As the name suggests, the `Dim` introduced with the `tuple` is a [tuple-like dimension](../DimensionKinds.md): it cannot be indexed using plain integers.
The value used for the index must be static. See the first section of [Dimension Kinds](../DimensionKinds.md).
(When a [traverser](../Traverser.md) is used, it detects this and uses the proper indexing.)

The size of a tuple is equal to the sum of the sizes of the [sub-structures](../Glossary.md#sub-structure). The tuple does not add any padding.


## Usage examples

While it is possible to combine tuples with other structures arbitrarily, there are two main usages: array of structures (AoS) and structure of arrays (SoA).

Consider for example a weighted graph (network) represented as a list of edges. Each edge has two endpoints (`int` IDs) and a weight/cost/capacity/... (`float`).
The simpler case is AoS:

```cpp
auto edges_aos = noarr::pack(noarr::scalar<int>(), noarr::scalar<int>(), noarr::scalar<float>()) ^ noarr::tuple<'t'>() ^ noarr::array<'i', 1024>();
```

It will be useful to extract the `array`:

```cpp
auto a = noarr::array<'i', 1024>(); // or vector if it should not be hardcoded
auto edges_aos = noarr::pack(noarr::scalar<int>(), noarr::scalar<int>(), noarr::scalar<float>()) ^ noarr::tuple<'t'>() ^ a;
```

Unfortunately, there is currently no way to similarly extract the tuple. Still the extraction of `a` helps.
Now it will be easier to move the array down, inside the tuple:

```cpp
auto edges_soa = noarr::pack(noarr::scalar<int>() ^ a, noarr::scalar<int>() ^ a, noarr::scalar<float>() ^ a) ^ noarr::tuple<'t'>();
```

The main advantage noarr tuples bring here is the layout agnosticity. The following algorithm works with both `edges_aos` and `edges_soa`:

```cpp
auto edges = noarr::make_bag(edges_soa, data_ptr); // or edges_aos

// a static index is necessary to index a tuple -- noarr::lit is a shortcut to create one
using noarr::lit;

for(std::size_t i = 0; i < num_edges; i++) {
	// the order of indices is not significant (but do not do this)
	int x = edges.template at<'t', 'i'>(lit<0>, i);
	int y = edges.template at<'i', 't'>(i, lit<1>);
	float force = edges.template at<'t', 'i'>(lit<2>, i);

	// a dimension can be fixed regardless of the layout
	// (only in AoS the result will be contiguous, but that does not matter much)
	auto edge = edges ^ noarr::fix<'i'>(i);
	int also_x = edge.template at<'t'>(lit<0>); // i already fixed, t remained
	// ...
}
```

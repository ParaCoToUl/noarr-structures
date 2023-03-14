# tuple

Concatenate multiple structure layouts and use a new [dimension](../Glossary.md#dimension)
to select between the original structures (now elements of the whole).

```hpp
#include <noarr/structures.hpp>

template<char Dim, typename... Ts>
struct noarr::tuple;
```


## Description

The layout of a `tuple` structure consists of the layouts of each of `Ts`, in order, one after another without any padding.
The whole structure has one additional dimension, named `Dim`. The [index](../Glossary.md#index) in `Dim` is used to select one of the `Ts`.

In order to use any of the [sub-structures](../Glossary.md#sub-structure) (`Ts`),
it is necessary to [set the index](../BasicUsage.md) in `Dim`, as otherwise, some operations would be ambiguous.
As the name suggests, the `Dim` introduced with the `tuple` is a [tuple-like dimension](../DimensionKinds.md): it cannot be indexed using plain integers.
The value used for the index must be static. See the first section of [Dimension Kinds](../DimensionKinds.md).

The size of a tuple is equal to the sum of the sizes of the [sub-structures](../Glossary.md#sub-structure). The tuple does not add any padding.


## Usage examples

Here are some of the valid tuple declarations:

```cpp
noarr::tuple<'t', noarr::scalar<int>, 
	noarr::scalar<float>> t;
noarr::tuple<'t', noarr::array<'x', 10, 
	noarr::scalar<float>>, noarr::vector<'x', 
	noarr::scalar<int>>> t2;
noarr::tuple<'t', noarr::array<'y', 20000, 
	noarr::vector<'x', noarr::scalar<float>>>,
	noarr::vector<'x', noarr::array<'y', 20, 
	noarr::scalar<int>>>> t3;
```

We will work with `tuple`s like this:

```cpp
// tuple declaration
noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, 
	noarr::array<'x', 20, noarr::scalar<int>>> tuple;
// we will create a bag
auto tuple_bag = noarr::make_bag(tuple);
// we have to use noarr::literals namespace 
	// to be able to index tuples
// we can put this at the beginning of the file
using namespace noarr::literals;
// we index tuple like this
// note that we fix multiple dimensions at one
float& value = tuple_bag.at<'t', 'x'>(0_idx, 1);
```

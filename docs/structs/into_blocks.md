# into_blocks

Split the structure into blocks according to the specified [dimension](../Glossary.md#dimension).

```hpp
#include <noarr/structures_extended.hpp>

template<char Dim, char DimMajor, char DimMinor, typename T>
struct noarr::into_blocks_t;

template<char Dim, char DimMajor, char DimMinor, char DimIsPresent, typename T>
struct noarr::into_blocks_dynamic_t;

template<char Dim, char DimIsBorder, char DimMajor, char DimMinor, typename T, typename MinorLenT>
struct noarr::into_blocks_static_t;

template<char Dim, char DimMajor, char DimMinor>
constexpr proto noarr::into_blocks();

template<char Dim, char DimMajor, char DimMinor>
constexpr proto noarr::into_blocks(auto minor_length);
// = noarr::into_blocks<Dim, DimMajor, DimMinor>() ^ noarr::set_length<DimMinor>(minor_length)

template<char Dim, char DimMajor, char DimMinor, char DimIsPresent>
constexpr proto noarr::into_blocks_dynamic();

template<char Dim, char DimMajor, char DimMinor, char DimIsPresent>
constexpr proto noarr::into_blocks_dynamic(auto minor_length);
// = noarr::into_blocks_dynamic<Dim, DimMajor, DimMinor, DimIsPresent>() ^ noarr::set_length<DimMinor>(minor_length)

template<char Dim, char DimIsBorder, char DimMajor, char DimMinor>
constexpr proto noarr::into_blocks_static(auto minor_length);
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

The simplest of these structures, `into_blocks_t` takes an existing dimension `Dim` of structure `T` and replaces it with a pair of dimensions, `DimMajor` and `DimMinor`.
The meaning of the new dimensions can then be summarized as follows:

- the [length](../Glossary.md#length) in `DimMajor` is the number of blocks
- the length in `DimMinor` is the block size (element count in one block)
- the [index](../Glossary.md#index) in `DimMajor` is the block index
- the index in `DimMinor` is the element index within the current block
- the index in the original `Dim` is computed as: index in `DimMajor` · length in `DimMinor` + index in `DimMinor`

By default, the block size (length in `DimMinor`) is not known and must be [set externally](../BasicUsage.md#lengths).
However, there are shortcuts that take a `minor_length` parameter.
See the first section of [Dimension Kinds](../DimensionKinds.md) for the allowed types of `minor_length`.

Note that the memory layout is not modified - only the view is changed.
Also note that `into_blocks_t` assumes the original length is a multiple of block size.
If that is not true, you should use `into_blocks_static_t` or `into_blocks_dynamic_t`.
These two handle incomplete blocks in two different ways.

### into_blocks_static

`into_blocks_static_t` first splits the structure in a "body" and a "border".
The length of body will be the largest multiple of block size that is not greater than the length in `Dim`.
In other words, it will be the original length, floored to the block size.
The remaining elements go to the border, so the border size is always (strictly) less than block size (and may be zero).
The border size is computed as the original length modulo block size.

A new dimension is added: `DimIsBorder`. This new dimension always has length `2` (thus the only valid indices are `0` and `1`).
Its index acts as a boolean flag: it is `0` in the body and `1` in the border.

*Only then*, the splitting into blocks is done on the body, as in `into_blocks_t`, that is, `Dim` is split into `DimMajor` and `DimMinor`.

For consistency, the same split is also done in the border. `DimMajor` has length `1` (the only valid index is `0`)
`DimMinor` has the length set according to the border size (and as such may be empty).

Unlike `into_blocks_t` and `into_blocks_dynamic_t`, the block size in `into_blocks_static_t` must be given immediately.

### into_blocks_dynamic

`into_blocks_dynamic_t` takes the opposite approach. It starts by splitting into blocks. It allows the last block to be incomplete.
However, to maintain constant block size (which is the main reason for using blocks in the first place), it makes the last block contain "too many" elements.

To avoid having elements that point past the allocated memory or into an unrelated piece of the structure, the `DimIsPresent` dimension is added.
Normally, the dimension has size `1`, meaning its only valid index is `0`. This is true in all elements of all complete blocks.
It is also true in some elements of the last block, namely those that still "fit" into the original length.
However, if there are any elements that would be past the original length, their `DimIsPresent` will be empty (i.e. length `0` and no valid index).

In case the original length is, in fact, a multiple of block size, all blocks will be complete and `DimIsPresent` will always be non-empty (unit).

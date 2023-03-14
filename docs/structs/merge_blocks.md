# merge_blocks

Opposite of [`into_blocks`](into_blocks.md): Merge the two specified [dimensions](../Glossary.md#dimension) into one dimension.

```hpp
#include <noarr/structures_extended.hpp>

template<char DimMajor, char DimMinor, char Dim, typename T>
struct noarr::merge_blocks_t;

template<char DimMajor, char DimMinor, char Dim>
constexpr proto noarr::merge_blocks();
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

Interpret `DimMajor` and `DimMinor` of the original structure `T` as two components of a single dimension `Dim`.
In other words, replace the former two dimensions with the latter and use the [index](../Glossary.md#index) in the latter to compute the indices for the former two.
The [length](../Glossary.md#length) of the new dimension is computed as the product of the two original dimensions.

The relation of the original and new dimensions can then be summarized as follows:

- the length in `DimMajor` is the number of blocks
- the length in `DimMinor` is the block size (element count in one block)
- the index in `DimMajor` is the block index
- the index in `DimMinor` is the element index within the current block
- the index in `DimMajor` is computed as: index in `Dim` divided by length in `DimMinor` (block size)
- the index in `DimMinor` is computed as: index in `Dim` modulo length in `DimMinor` (block size)

Note that the memory layout is not modified -- only the view is changed.

As indicated above, `noarr::merge_blocks` could be considered a counterpart to `noarr::into_blocks`.
Conceptually, `merge_blocks<'M', 'm', 'f'>() ^ into_blocks<'f', 'M', 'm'>(???)`
and `into_blocks<'f', 'M', 'm'>(???) ^ merge_blocks<'M', 'm', 'f'>()` should both be noops.
(In practice, the choice of block size and the divisibility by it slightly complicate matters.)

In general, `merge_blocks` can be used to define advanced layouts (e.g. tiled) with simple interfaces,
while `into_blocks` can be used to traverse the simple layouts (like row-major or column-major) in a sophisticated way.

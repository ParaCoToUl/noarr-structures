# merge_zcurve

Merge the specified [dimensions](../Glossary.md#dimension) into one dimension
which walks the original dimensions in the [Z-order curve](https://en.wikipedia.org/wiki/Z-order_curve).

```hpp
#include <noarr/structures/structs/zcurve.hpp>

template<char... Dims, char Dim>
struct noarr::merge_zcurve {
	template<int MaxLen, int Alignment>
	constexpr proto maxlen_alignment();
};
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

This structure folds several (usually two or three) dimensions of the original structure (`Dims`) into one dimension (`Dim`).
As such, the original `Dims` are missing from the newly created structure, they are replaced with `Dim`.

As always, the resulting dimension does not contain any gaps (and its [length](../Glossary.md#length) is the product of lengths in all `Dims`).
Although the Z-order curve is usually defined only for dimensions of the same length (which must additionally be a power of two),
this structure lifts this restriction: the `Dims` can be of any lengths.

This however complicates the algorithm, which could negatively impact the performance.
If you know in advance that the lengths of all `Dims` are powers of two (or at least multiples of some power of two other than one),
you can adjust the parameters to use a more efficient (and less general) algorithm. Likewise if the length has an upper bound.

The `MaxLen` parameter is the maximum length in the longest of `Dims`.
Set it to `1 << (SIZE_WIDTH - 1)` if you cannot promise any bound (this is half the range of `size_t`; `merge_zcurve` does not support more).
This parameter must be a power of two -- if the maximum length is not, use the next *larger* power of two.

Set the `Alignment` parameter to the largest power of two such that all the lengths in all `Dims` will always be multiples of `Alignment`.
If you cannot promise any bound, set this to one (since all integers are divisible by one).
If you know that the length will always be a power of two (just don't know which one), set this to the same value as `MaxLen`.

This structure is intended to guide traversals of the usual multidimensional structures (e.g. row-major or column-major).
This means it is suitable for the implementation of cache-oblivious algorithms such as those for matrix transposition or multiplication.
It cannot be used to store multidimensional into memory using z-order curve (as is sometimes done e.g. with textures or volumetric data).
For that, an opposite of `merge_zcurve` would be necessary: ~~`into_zcurve`~~, which as of now is not implemented.
Alternatively, you can use a tiled format, created using [`noarr::merge_blocks`](merge_blocks.md).

One final note: unless your algorithm depends on it functionally, you should not use `merge_zcurve` all the way down to the individual elements:
Instead, split the structure [into blocks](into_blocks.md) and only use `merge_zcurve` on the block index (`DimMajor`),
leaving the element index (`DimMinor`) "untangled".

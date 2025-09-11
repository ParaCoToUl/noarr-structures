# merge_blocks

Opposite of [`into_blocks`](into_blocks.md): Merge the two specified [dimensions](../Glossary.md#dimension) into one dimension.

```hpp
#include <noarr/structures_extended.hpp>

template<auto DimMajor, auto DimMinor, auto Dim, typename T>
struct noarr::merge_blocks_t;

template<auto DimMajor, auto DimMinor, auto Dim>
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


## Usage examples

`noarr::merge_blocks` is used during data modeling. Say we want a structure that looks like the following:

```cpp
std::size_t num_rows = 8;
std::size_t num_cols = 12;

// i is row index, j is column index
auto simple_matrix = noarr::scalar<float>()
	^ noarr::vector<'j'>(num_cols)
	^ noarr::vector<'i'>(num_rows);
```

Like this, it will be stored row-by-row. But we might want it to be stored as vertical blocks of width 4.
That is, the layout will consist of `num_cols/4` blocks, each of which is `num_rows` rows of `4` elements.
That is 3 layers (block `'b'`, row `'i'`, elem `'e'`), so we need to start with a 3D structure:

```cpp
auto blocked_matrix = noarr::scalar<float>()
	^ noarr::vector<'e'>(4)
	^ noarr::vector<'i'>(num_rows)
	^ noarr::vector<'b'>(num_cols / 4);
```

But now, the structure will not have the interface we wanted. It requires its user to pass `'b'` and `'e'` instead of `'j'`.
Now it is time to use `merge_blocks` to hide this detail in the structure:

```cpp
auto final_matrix = blocked_matrix
	^ noarr::merge_blocks<'b', 'e', 'j'>();
```

The resulting `final_matrix` can be used in the same way as `simple_matrix` (i.e. it has dimensions `'i'` for row and `'j'` for column),
it will just use different offset calculation.

### Incomplete blocks

The above solution does not work with `num_cols` which is not a multiple of the block size (4).
One possible way to fix this is to round the number of columns up to a multiple of 4,
then convert to blocks, and only then apply the proper size:

```cpp
std::size_t block_size = 4;
std::size_t num_blocks = (num_cols + block_size - 1) / block_size; // This is ceiling(num_cols / block_size)

auto matrix = noarr::scalar<float>()
	^ noarr::vector<'e'>(block_size)
	^ noarr::vector<'i'>(num_rows)
	^ noarr::vector<'b'>(num_blocks)
	^ noarr::merge_blocks<'b', 'e', 'j'>()
	^ noarr::slice<'j'>(0, num_cols);
```

In the last step, we used [`noarr::slice`](slice.md) to view just the elements we want in the matrix
(in other words, to hide the elements added by the rounding).
Slicing does not change the size or layout, it just changes the apparent length in `'j'`.

### Tiled layout

This transformation can of course be applied repeatedly. In order to avoid confusion,
it might be useful to name the block index `'J'` and the index-within-block `'j'`, to show the relation to the `'j'` in `simple_matrix`.
Now we can do the same with `'i'`:

```cpp
std::size_t tile_rows = 4;
std::size_t tile_cols = 4;

auto tiled_matrix = noarr::scalar<float>()
	// Create a tile
	^ noarr::vector<'j'>(tile_cols)
	^ noarr::vector<'i'>(tile_rows)
	// Create the grid of tiles
	^ noarr::vector<'J'>(num_cols / tile_cols)
	^ noarr::vector<'I'>(num_rows / tile_rows)
	// Merge the tiles together
	^ noarr::merge_blocks<'J', 'j', 'j'>()
	^ noarr::merge_blocks<'I', 'i', 'i'>();
```

The order of the last two transformations is not significant.
The same trick with [incomplete blocks](#incomplete-blocks) can be used here, whether for both dimensions or only for one of them.

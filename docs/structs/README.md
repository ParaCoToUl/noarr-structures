# Built-in structures

- [`scalar`](scalar.md): the simplest possible structure referring to a single value of a certain type
- [`tuple`](tuple.md): introduces a static dimension with many substructures (similar to `std::tuple`)
- [`vector`](vector.md): introduces a dynamic dimension of an unknown length (has to be specified ad-hoc) or a known (dynamic/static) length
- [`array`](array.md): introduces a dynamic dimension of a known static length (using `noarr::vector`)
- [`fix`](fix.md): fixes an index in a structure
- [`set_length`](set_length.md): sets the length (number of indices) of a structure in the given dimension
- [`hoist`](hoist.md): selects one dimension by its name and moves it to the top level
- [`rename`](rename.md): assigns different names to zero or more dimensions (swapping names is allowed)
- [`shift`](shift.md): makes the specified dimension start at the specified index, making the prefix of each row/column inaccessible
- [`slice`](slice.md): makes the specified dimension start at some index and end at another index, making a prefix and a suffix inaccessible
- [`into_blocks`](into_blocks.md): splits one dimension into two dimensions, one of which becomes the index of a block, and the other the index within a block
- [`merge_blocks`](merge_blocks.md): the inverse of `into_blocks` - takes two existing dimensions and merges them into one dimension, making one of the original dimensions the index of a block and the other the index within a block
- [`merge_zcurve`](merge_zcurve.md): like `merge_blocks`, but does not compose the dimensions using blocks but a z-order curve instead (this structure also supports any number of dimensions, not just two)
- [`bcast`](bcast.md): introduces a dynamic dimension that is ignored
- [`step`](step.md): selects every (a+bi)th element according to the specified dimension
- [`cuda_step`](cuda_step.md): splits a structure among cuda threads (using `noarr::step`)
- [`cuda_striped`](cuda_striped.md): creates multiple copies (stripes) of a structure, each to be used by only some threads

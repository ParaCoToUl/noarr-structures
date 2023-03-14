# Built-in structures

- `scalar`: the simplest possible structure referring to a single value of a certain type
- `tuple`: introduces a static dimension with many substructures (similar to `std::tuple`)
- `array`: introduces a dynamic dimension of a static length with a single substructure (similar to `std::array`)
- `vector`: introduces a dynamic dimension of a dynamic length (has to be specified ad-hoc) with a single substructure (similar to `std::vector, .resize`)
- `fix`: fixes an index in a structure
- `set_length`: changes the length (number of indices) of arrays and vectors
- `reorder`: reorders the dimensions according to the specified list of dimension names. Dimensions may be omitted.
- `hoist`: selects one dimension by its name and moves it to the top level
- `rename`: assigns different names to zero or more dimensions. Swapping names is allowed.
- `shift`: makes the specified dimension start at the specified index, making the prefix of each row/column inaccessible
- `slice`: makes the specified dimension start at some index and end at another index, making a prefix and a suffix inaccessible
- `into_blocks`: splits one dimension into two dimensions, one of which becomes the index of a block, and the other the index within a block
- `merge_blocks`: the inverse of `into_blocks` - takes two existing dimensions and merges them into one dimension, making one of the original dimensions the index of a block and the other the index within a block
- `merge_zcurve`: like `merge_blocks`, but does not compose the dimensions using blocks but a z-order curve instead. This structure also supports any number of dimensions, not just two

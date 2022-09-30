# Noarr Structures file structure outline

- [funcs.hpp](funcs.hpp): contains implementations for functions (applicable in the piping mechanism, see the description for [pipes.hpp](pipes.hpp) defined by the library.

  - `operator|`: applies a function (the right-hand operand) to a structure (the left-hand operand), left-associative
  - `get_length`: gets the length (number of indices) of a structure
  - `get_size`: returns the size of the data represented by the structure in bytes
  - `offset`: retrieves offset of a substructure (if no substructure is specified it defaults to the innermost one: the scalar value), allows for ad-hoc fixing of dimensions
  - `get_at`: returns a reference to a value in a given blob the offset of which is specified by a dimensionless (same as `offset`) structure, allows for ad-hoc fixing of dimensions

- [contain.hpp](contain.hpp): contains the implementation of the `contain` struct. It mimics the basic functionality of `std::tuple`. It facilitates creating structures (and other constructs like functions) so they are *trivial structured layouts* and so there is a structured way of retrieving fields.

  - `contain`: tuple-like struct, and a structured layout, that facilitates creation of new structures and functions

- [contain_serialize.hpp](contain_serialize.hpp): contains implementations for `serialize` and `deserialize`, functions that serialize and deserialize structures

- [scalar.hpp](scalar.hpp): contains the implementation of the `scalar` structure. It serves as the bottom-most node for other structures (e.g. `vector<'v', scalar<int>>`).

  - `scalar`: the simplest possible structure referring to a single value of a certain type

- [structs.hpp](structs.hpp): contains implementations of the structures defined by the library (other than `scalar`).

  - `tuple`: introduces a static dimension with many substructures (similar to `std::tuple`)
  - `array`: introduces a dynamic dimension of a static length with a single substructure (similar to `std::array`)
  - `vector`: introduces a dynamic dimension of a dynamic length (has to be specified ad-hoc) with a single substructure (similar to `std::vector, .resize`)

- [setters.hpp](structs.hpp): contains structures that fix free variables in the structure definition

  - `fix`: fixes an index in a structure
  - `set_length`: changes the length (number of indices) of arrays and vectors

- [reorder.hpp](structs.hpp): contains structures that change the signature of a structure without changing its layout

  - `reorder`: reorders the dimensions according to the specified list of dimension names. Dimensions may be omitted.
  - `hoist`: selects one dimension by its name and moves it to the top level

- [view.hpp](structs.hpp): contains structures that change the way each dimension is accessed from outside

  - `rename`: assigns different names to zero or more dimensions. Swapping names is allowed.
  - `shift`: makes the specified dimension start at the specified index, making the prefix of each row/column inaccessible
  - `slice`: makes the specified dimension start at some index and end at another index, making a prefix and a suffix inaccessible

- [blocks.hpp](structs.hpp): contains structures that compose or decompose existing dimensions into blocks

  - `into_blocks`: splits one dimension into two dimensions, one of which becomes the index of a block, and the other the index within a block
  - `merge_blocks`: the inverse of `into_blocks` - takes two existing dimensions and merges them into one dimension, making one of the original dimensions the index of a block and the other the index within a block

- [zcurve.hpp](structs.hpp): contains structures that compute indices using the z-order curve

  - `merge_zcurve`: like `merge_blocks`, but does not compose the dimensions using blocks but a z-order curve instead. This structure also supports any number of dimensions, not just two

- [wrapper.hpp](wrapper.hpp): contains the implementation of the `wrapper` struct. It encapsulates an arbitrary structure and allows for the use of the dot notation for applying functions instead of the pipe notation.

  - `wrapper`: a wrapper for a structure, allows dot notation instead of using `operator|` (see `pipes.hpp`)
  - `wrap`: wraps a structure inside a `wrapper`

- [bag.hpp](bag.hpp): contains the implementation of the `bag` struct. It creates a data structure with an underlying blob containing values described by the layout of the "bagged" structure.

  - `bag`: a simple class that contains a wrapped structure and an underlying blob

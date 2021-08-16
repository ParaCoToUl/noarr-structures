# Noarr Structures file structure outline

- [pipes.hpp](pipes.hpp): contains the implementation for the piping mechanism of the library. Piping is a left-associative operation which applies the right-hand operand (a function) to the left-hand operand (generally a structure)

  - `operator|`: applies a function (the right-hand operand) to a structure (the left-hand operand), left-associative
  - `pipe`: chains its operands in multiple applications using left-associative `operator|`

- [funcs.hpp](funcs.hpp): contains implementations for functions (applicable in the piping mechanism, see the description for [pipes.hpp](pipes.hpp) defined by the library.

  - `compose`: function composition (honoring the left-associative `|` notation)
  - `set_length`: changes the length (number of indices) of arrays and vectors
  - `get_length`: gets the length (number of indices) of a structure
  - `get_size`: returns the size of the data represented by the structure in bytes
  - `fix`: fixes an index in a structure
  - `get_offset`: retrieves offset of a substructure
  - `offset`: retrieves offset of a value in a structure with no dimensions (or in a structure with all dimensions being fixed), allows for ad-hoc fixing of dimensions
  - `get_at`: returns a reference to a value in a given blob the offset of which is specified by a dimensionless (same as `offset`) structure, allows for ad-hoc fixing of dimensions

- [contain.hpp](contain.hpp): contains the implementation of the `contain` struct. It mimics the basic functionality of `std::tuple`. It facilitates creating structures (and other constructs like functions) so they are *trivial structured layouts* and so there is a structured way of retrieving fields.

  - `contain`: tuple-like struct, and a structured layout, that facilitates creation of new structures and functions

- [scalar.hpp](scalar.hpp): contains the implementation of the `scalar` structure. It serves as the bottom-most node for other structures (e.g. `vector<'v', scalar<int>>`).

  - `scalar`: the simplest possible structure referring to a single value of a certain type

- [structs.hpp](structs.hpp): contains implementations of the structures defined by the library (other than `scalar`).

  - `tuple`: introduces a static dimension with many substructures (similar to `std::tuple`)
  - `array`: introduces a dynamic dimension of a static length with a single substructure (similar to `std::array`)
  - `vector`: introduces a dynamic dimension of a dynamic length (has to be specified ad-hoc) with a single substructure (similar to `std::vector, .resize`)

- [wrapper.hpp](wrapper.hpp): contains the implementation of the `wrapper` struct. It encapsulates an arbitrary structure and allows for the use of the dot notation for applying functions instead of the pipe notation.

  - `wrapper`: a wrapper for a structure, allows dot notation instead of using `operator|` (see `pipes.hpp`)
  - `wrap`: wraps a structure inside a `wrapper`

- [bag.hpp](bag.hpp): contains the implementation of the `bag` struct. It creates a data structure with underlying blob containing values described by the layout of the "bagged" structure.

  - `bag`: simple class that contains a wrapped structure and an underlying blob

# Noarr structures

## The file structure

- [bag.hpp](bag.hpp)
  - `bag`: simple class that contains a `wrap`ped structure and an underlying blob
- [contain.hpp](contain.hpp)
  - `contain`: tuple-like struct, and a structured layout, that facilitates creation of new structures and functions
- [core.hpp](core.hpp)
  - `operator|`: applies a function (the right-hand operand) to a structure (the left-hand operand), left-associative
  - `pipe`: chains its operands in multiple applications using left-associative `operator|`
- [funcs.hpp](funcs.hpp)
  - `compose`: function composition (honoring the left-associative notation)
  - `set_length`: changes the length (number of indices) of arrays and vectors
  - `get_length`: gets the length (number of indices) of a structure
  - `fix`: fixes an index in a structure
  - `get_offset`: retrieves offset of a substructure 
  - `offset`: retrieves offset of a value in a structure with no dimensions (or in a structure with all dimensions being fixed), allows for ad-hoc fixing of dimensions
  - `get_at`: returns a reference to a value in a given blob the offset of which is specified by a dimensionless (same as `offset`) structure, allows for ad-hoc fixing of dimensions
- [scalar.hpp]
  - `scalar`: the simplest possible structure referring to a single value
- [structs.hpp](structs.hpp)
  - `tuple`: introduces a static dimension with many substructures (similar to `std::tuple`)
  - `array`: introduces a dynamic dimension of a static length with a single substructure (similar to `std::array`)
  - `vector`: introduces a dynamic dimension of a dynamic length (has to be specified ad-hoc) with a single substructure (similar to `std::vector, .resize`)
- [wrapper.hpp](wrapper.hpp)
  - `wrapper`: a wrapper for a structure, allows dot notation instead of using `operator|` (see `core.hpp`)
  - `wrap`: wraps a structure inside a `wrapper`

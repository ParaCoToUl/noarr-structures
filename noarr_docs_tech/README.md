# Technical specification

## Structure

Structure is a simple object that describes data layouts and their abstractions

### Structure requirements

For a structure `T`:

- the expression `std::is_trivial<T>::value && std::is_standard_layout<T>::value` shall evaluate to `true` (= it shall be a *PODType*)
- `T::sub_structures()` returns a tuple of sub-structures that can be (replaced and) used to `construct` a new structure `T`
  - it shall be a *pure function*
  - it shall be `constexpr` and either `static` or `const`. It shall be `static` iff the structure has no sub-structure
    - *example of a structure with no sub-structure: `scalar<T>` (serves as a ground (or leaf) structure)*
  - it shall satisfy `consteval`
- `T::construct(T1, T2, ...)` creates a new structure `T` from sub-structures `T1`, `T2`, ...
  - it shall take the same universe of sub-structures that `T::sub_structures()` returns in tuple, and the `construct`ing structure shall be identical to the structure `construct`ed from it given these sub-structures as arguments
  - it shall be `constexpr` and either `static` or `const`. It shall be `static` iff the structure depends solely on its sub-structures
    - *e.g. `array::construct` is `static`, but `sized_vector::construct` is `const` as its length is not dependent on its sub-structures*
  - if `T::construct` is `const`:
    - each `construct`ed structure shall be identifiable by its sub-structures and a *prototype*. A prototype is a structure whose `construct` method, given the sub-structures as arguments, produces the aforementioned constructed structure
    - it shall follow that the relationship of having/being a prototype is a mathematical equivalence (transitive, reflexive and symmetric)
      - and thus the structures can be divided into equivalence classes where any structure from a certain class can be called a prototype of the whole class (these equivalence classes don't have to correspond with generic `class`es defining the structures)
        - *e.g. `sized_vector`s can be divided into equivalence classes by their length*
  - if `T::construct` is `static`
    - the `construct`ed structure is identifiable by its sub-structures and thus the relationship of being a *prototype* (defined as above) forms a single equivalence class (again, do not mix up with `class`es as defined by the programming language)
  - it shall satisfy `consteval` if it is allowed by the semantics of the structure (possible exceptions: structures implemented for debugging purposes)
  - it shall be a *pure function* if it is allowed by the semantics of the structure (possible exceptions: structures implemented for debugging purposes)
- the structure shall define `description`, a type that is a specialization of `struct_description` and describes the structure
  - the first entry shall be a `char_pack` specialization containing the structure's name
  - the second entry shall be a `dims_impl` specialization containing the dimension (if any - *e.g. `scalar<T>` doesn't introduce a dimension*) the structure introduces
  - the third entry shall be a `dims_impl` specialization containing the dimensions (if any) the structure consumes from its sub-structures
  - the other entries are each a specialization of either `struct_param` or `struct_param` <!-- TODO -->
- `T::length()` is a function that returns a `std::size_t` value which specifies the range of indices the structure supports via `T::offset`
  - it shall be `constexpr`
  - it shall be either `static` or `const`
  - if the structure has no dimension it shall be `static` and return `0`
- `T::offset()`, `T::offset<std::size_t>()`, or `T::offset(std::size_t)` is a function that returns a `std::size_t` value which specifies the offset of a (sub-structure) instance with the given index
  - it shall be `constexpr`
  - it shall be either `static` or `const`
  - it shall take an argument (either template or formal) iff the structure has a dimension
  - the implementation of `T::offset` should satisfy requirements for it implied by requirements for `T::get_t`
  - if a positive integer `n` is a valid argument, then the argument `n - 1` shall be valid as well
  - the number of all valid arguments for `T::offset` shall be equal to `T::length()`
    - *as a result, if `l = T::length()` is positive then `l - k` will be a valid argument for `T::offset`; where `k` is an integer s.t. `0 < k < l + 1`*
- `T::get_t<...>` is a type of the (sub-structure) instance at `T::offset` given the following:
  - if the structure has no dimension then `T::get_t` shall take both no argument or a single `void` argument, `T::offset` shall take no argument, and they shall return the type and the offset of the only (sub-structure) instance , respectively
    - `T::get<>` and `T::get<void>` shall return the same type
  - if the structure has one static dimension then `T::get_t` shall take a single `std::integral_constant<std::size_t, ...>` argument, `T::offset` shall take one `std::size_t` template argument , and they shall return the type and the offset of the (sub-structure) instance with the given index
  - if the structure has one dynamic dimension then `T::get_t` shall combine the behavior of the `get_t` of a structure with no dimension and of a structure with one static dimension, `T::offset` shall take both one `std::size_t` formal argument or one `std::size_t` template argument, and they shall return the type and the offset of the (sub-structure) instance with the given index
    - `T::get_t` shall return the same type for any correct arguments given
    - `T::offset` shall return the same offset for the given index regardless of whether it was given via template or formally
  - if the structure has a dimension then it is called static or dynamic if the structure would satisfy the previous statements (this means that the staticity or dynamicity of the dimension is deduced)
  - if the structure has a dimension then it shall be either static or dynamic
  - if the structure has any sub-structures, `T::get_t` shall always return one of types of sub-structures in `T::sub_structures()` and for every sub-structure type there shall exist an argument to `T::get_t` such that it returns this sub-structure and they shall share their indices
    - *as a result, structures with a dynamic dimension will have just one sub-structure or it should be a ground (leaf) structure*
  - if the structure has no sub-structures, `T::get_t` shall return a representation of the type of the physical data

### Cube

A cube is a structure hierarchy which has all its dimensions dynamic. This has a consequence of having a single scalar type

### Point

A point is a structure hierarchy with no dimensions. It has only a single scalar type. It is a special case of cube.


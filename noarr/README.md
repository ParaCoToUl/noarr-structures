# Documentation

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

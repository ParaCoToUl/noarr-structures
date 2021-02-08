# Documentation

## Struct requirements

For a structure `T`:

- the expression `std::is_trivial<T>::value && std::is_standard_layout<T>::value` shall evaluate to `true` (= it shall be a *PODType*)
- `T::sub_structures()` returns a tuple of sub-structures that can be (replaced and) used to create a new structure `T`
  - it shall be `constexpr` and either `static` or `const` (it shall be `static` iff the structure has no sub-structure)
  - it shall be a pure function
  - it shall satisfy `consteval`
- `T::construct(T1, T2, ...)` takes the same sub-structures that `T::sub_structures()` returns in the tuple, this creates the new structure `T` and it shall be indistinguishable from the original structure if none of the sub-structures was changed or if all changed sub-structures are indistinguishable from the sub-structures of the original structure
  - it shall be `constexpr` and either `static` or `const` (it shall be `static` iff the structure depends solely on its sub-structures - e.g. `sized_vector::construct` is `const` and `array::construct` is `static`)
  - it shall be a pure function
  - all structures constructed from the same structure and having indistinguishable sub-structures shall be indistinguishable
  - A new structure constructed from structures constructed from a structure `T t` is indistinguishable from a structure constructed from the structure `t` if these two constructions took indistinguishable sub-structures
  - *two structures can have indistinguishable sub-structures and still be distinguishable from each other: e.g. two `sized_vector`s of a different length*
- the structure shall define `description`, a type that is a specialization of `struct_description` and describes the struct
  - the first entry shall be a `char_pack` specialization containing the structure's name
  - the second entry shall be a `dims_impl` specialization containing the dimension (if any - *e.g. `scalar<T>` doesn't introduce a dimension*) the structure introduces
  - the third entry shall be a `dims_impl` specialization containing the dimensions (if any) the structure consumes from its sub-structures
  - the other entries are each a specialization of either `struct_param` or `struct_param` <!-- TODO -->

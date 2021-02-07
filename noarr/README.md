# Documentation

## Struct requirements

For a structure `T`:

- the expression `std::is_trivial<T>::value && std::is_standard_layout<T>::value` has to evaluate to `true` (= it has to be a *PODType*)
- `T::sub_structures()` returns a tuple of sub-structures that can be (replaced and) used to create a new structure `T`
  - it has to be `constexpr` and either `static` or `const`
  - it has to be a pure function
  - it has to satisfy `consteval`
- `T::construct(T1, T2, ...)` takes the same sub-structures that `T::sub_structures()` returns in the tuple, this creates the new structure `T` and it shall be indistinguishable from the original structure if none of the sub-structures was changed or if all changed sub-structures are indistinguishable from the sub-structures of the original structure
  - it has to be `constexpr` and either `static` or `const`
  - it has to be a pure function
  - all structures constructed from the same structure and having indistinguishable sub-structures shall be indistinguishable
  - A new structure constructed from a structures constructed from a structure `T t` is indistinguishable from a structure constructed from the structure `t` if these two constructions took indistinguishable sub-structures
  - *two structures can have indistinguishable sub-structures and still be distinguishable from each other: e.g. two vectors of a different length*

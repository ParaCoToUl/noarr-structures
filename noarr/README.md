# Documentation

## Struct requirements

For a structure `T`:

- the expression `std::is_trivial<T>::value && std::is_standard_layout<T>::value` has to evaluate to `true` (= it has to be a *PODType*)
- `T::sub_structures()` returns a tuple of sub-structures that can be (replaced and) used to create a new structure `T`
  - it has to be `constexpr` and either `static` or `const`
- `T::construct(T1, T2, ...)` takes the same sub-structures that `T::sub_structures()` returns in the tuple, this creates the new structure `T`
  - it has to be `constexpr` and either `static` or `const`

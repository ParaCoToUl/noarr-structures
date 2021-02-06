# Documentation

## Struct requirements

For a structure `T`:

- the standard `std::is_trivial<T>::value` has to evaluate to `true`
- `T::sub_structures()` returns a tuple of sub-structures that can be (replaced and) used to create a new structure `T`
  - it has to be `constexpr` and either `static` or `const`
- `T::construct(T1, T2, ...)` takes the same sub-structures that `T::sub_structures()` returns in the tuple, this creates the new structure `T`
  - it has to be `constexpr` and either `static` or `const`

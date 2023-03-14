# Technical documentation for Noarr Structures

## Structure

A  *structure* is a simple object that describes data layouts and their abstractions

### Structure requirements

For a structure `T`:

- the expression `std::is_trivial<T>::value && std::is_standard_layout<T>::value` shall evaluate to `true` (= it shall be a *[PODType](https://en.cppreference.com/w/cpp/named_req/PODType)*), furthermore, it shall not define any fields in its body
  - all desired fields shall be defined by inheriting the tuple-like `contain` (see below)
- it shall inherit from `contain`
- the structure shall define `description`, a type that is an instance of `struct_description`, and describes the structure and its type parameters
  - the first entry shall be a `char_sequence` instance containing the structure's name
  - the second entry shall be a `dims_impl` instance containing the dimension (if any - *e.g. `scalar<T>` does not introduce a dimension*) the structure introduces
  - the third entry shall be a `dims_impl` instance containing the dimensions (if any) the structure consumes from its sub-structures
  - the other entries shall each be a either `structure_param` or `type_param`, `type_param` for (scalar) type parameters and `structure_param` for types that represent structures
  - this `description` should be implemented in such a way that `print_struct(std::ostream&, structure)` outputs an equivalent of the structure type with all type parameters written in C++ (this is not a technical requirement, but not satisfying it can hinder any data serialization validation based on `print_struct`)
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

### Subtypes of structures

- **Cube:** a cube is a structure hierarchy that has all its dimensions dynamic. This has a consequence of having a single scalar type (all values described by the structure share the same type).

- **Point:** a point is a structure hierarchy with no dimensions. It has only a single scalar type and it describes one scalar value of this type.

  It is a special case of a cube.

### Contain

`contain` facilitates the creation of new structures. It is a tuple-like struct that defines a struct's fields via inheritance, but in contrast with `std::tuple`, it is a trivially constructible standard layout.

It is used in the library to define all structures (and the majority of all noarr functions). This is to provide a structured way of serializing a structure's data.

### Provided structures

The library provides the following set of structures that describe the most essential layouts:

- **`scalar`:** contains a single scalar type and describes one value of this type. It serves as the leaf substructure in structure hierarchies and as the bottom case for many algorithms and mechanisms defined by the library. Its size is equal to the size of the contained type and it is always known during compile time.
- **`array`:** a structure containing a single substructure and providing a dynamic dimension. The layout described by an array consists of a static number of copies of the layout described by the contained substructure lined up right after one another. Its size is equal to the size of the contained substructure multiplied by the number of its copies and it is always known during compile time if the size of the substructure is as well.
- **`vector`:** a structure containing a single substructure and providing a dynamic dimension. It is very similar to array (see above) with the distinction that the number of the substructure's layout copies is dynamic, and because of it being dynamic, the size of the vector is dynamic as well and it is generally not known during compile time.
- **`tuple`:** a structure containing multiple substructures and providing a static dimension. It describes a layout consisting of the layouts of the substructures lined up one after another. Its size is equal to the sum of the sizes of the substructures and it is known during compile time if all sizes of the substructures are as well.

## High-level abstractions and utilities

The library provides various high-level abstractions and utilities that either simplify the usage of the lower-level structures and functions, or they expand on them

### Bag

`bag` combines a wrapped structure with an underlying data blob.

It provides the following extra methods:

- `data()` returns the underlying data blob
- `structure()` returns the wrapped structure
- `at<Dims...>(values...)` returns a reference to a value in the data blob with the offset computed fixing dimensions named `Dims...` (`char` constants) to `values...` respectively. In other words it is a shortcut for `bag.structure() | get_at<Dims...>(bag.data(), values...)`

There are various types of `bag`s:

- vector bag

  - created by `make_vector_bag(structure)`
  - the underlying data blob is implemented via a `std::vector`
  - the bag destroys the data blob in its destructor

- unique bag (the default type)

  - created by `make_unique_bag(structure)` or `make_bag(structure)`
  - the underlying data blob is implemented via a `std::unique_ptr`
  - the bag destroys the data blob in its destructor

- observer bag

  - created by `make_bag(structure, char_carray)`
  - the underlying data blob is implemented via `char *`
  - destruction of bag does not affect the data blob

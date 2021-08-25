# Technical documentation for Noarr Structures

<!-- TODO: list all functions and what they do... -->

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
      - and thus the structures can be divided into equivalence classes where any structure from a certain class can be called a prototype of the whole class (these equivalence classes do not have to correspond with generic `class`es defining the structures)
        - *e.g. `sized_vector`s can be divided into equivalence classes by their length*
  - if `T::construct` is `static`
    - the `construct`ed structure is identifiable by its sub-structures and thus the relationship of being a *prototype* (defined as above) forms a single equivalence class (again, do not mix up with `class`es as defined by the programming language)
  - it shall satisfy `consteval` if it is allowed by the semantics of the structure (possible exceptions: structures implemented for debugging purposes)
  - it shall be a *pure function* if it is allowed by the semantics of the structure (possible exceptions: structures implemented for debugging purposes)
- the structure shall define `description`, a type that is a specialization of `struct_description` and describes the structure
  - the first entry shall be a `char_pack` specialization containing the structure's name
  - the second entry shall be a `dims_impl` specialization containing the dimension (if any - *e.g. `scalar<T>` does not introduce a dimension*) the structure introduces
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

### Subtypes of structures

- **Cube:** a cube is a structure hierarchy which has all its dimensions dynamic. This has a consequence of having a single scalar type (all values described by the structure share the same type).

- **Point:** a point is a structure hierarchy with no dimensions. It has only a single scalar type and it describes one scalar value of this type.

  It is a special case of cube.

### Contain

`contain` facilitates creation of new structures. It is a tuple-like struct that defines a struct's fields via inheritance, but in contrast with `std::tuple`, it is a trivially constructible standard layout.

It is used in the library to define all structures (and the majority of all noarr functions). This is to provide a structured way of serializing a structure's data.

### Provided structures

The library provides the following set of structures that describe the most essential layouts:

- **`scalar`:** contains a single scalar type and describes one value of this type. It serves as the leaf substructure in structure hierarchies and as the bottom case for many algorithms and mechanisms defined by the library. Its size is equal to the size of the contained type and it is always known during compile time.
- **`array`:** a structure containing a single substructure and providing a dynamic dimension. The layout described by an array consists of a static number of copies of the layout described by the contained substructure lined up right after one another. Its size is equal to the size of the contained substructure multiplied by the number of its copies and it is always know during compile time if the size of the substructure is as well.
- **`vector`:** a structure containing a single substructure and providing a dynamic dimension. It is very similar to array (see above) with the distinction that the number of the substructure's layout copies is dynamic, and because of it being dynamic, the size of vector is dynamic as well and it is generally not know during compile time.
- **`tuple`:** a structure containing multiple substructures and providing a static dimension. It describes a layout consisting of the layouts of the substructures lined up one after another. Its size is equal to the sum of the sizes of the substructures and it is known during compile time if all sizes of the substructures are as well.

### Helper structures

- **`sfixed_dim`:** eliminates the static (or dynamic) dimension of the contained structure by fixing a certain index in it while preserving the layout of the structure and all substructures.
- **`fixed_dim`:** eliminates the dynamic dimension of the contained structure by fixing a certain index in it while preserving the layout of the structure and all substructures.

## Noarr Function

Functions are (using the `operator|`) applied to structures. Applying them returns either another structure (this is mostly the case of the functions with `func_family` set to `transform_tag`, more on `func_family`s later in this section) or a scalar value (usually if `func_family` is set to `get_tag`).

### Piping

The piping mechanism (used inside `operator|`) is split into three cases:

- **Top application:** This case applies to the functions have their `func_family` set to `top_tag`.

  It is the simplest piping mechanism case as it is equivalent to simple application (e.g.: the expression `s | f`, if `f` has a `top_tag`, is equivalent to `f(s)`).

- **Getting:** This case applies to the functions have their `func_family` set to `get_tag`.

  It is an extension of the *top application* case:

  given the expression `s | f`, if `f(s)` is not a valid expression, the piping mechanism attempts to apply `f` to the substructures of `s` (recursively). It fails if `f` is not applicable to any of the substructures or if it is applicable to more substructures. In other words, it succeeds if and only if there is one and only one sub-graph branch in the structure hierarchy (if represented by a tree) such that `f` is not applicable to any of the non-leaf nodes of the branch and it is applicable to the leaf.

- **Transformation (mapping)**

  This case applies to the functions have their `func_family` set to `transform_tag`.

  The result of `s | f` results in applying `f` to the top-most structure of each branch of the structure hierarchy (or leaving the branch without change if `f` is applicable to none of the structures) and then reconstructing the structure with these changes to the substructures.

### Function requirements

Functions have rather informal requirements so the library can reach maximum expressiveness without affecting its complexity.

It is very preferable that each function honors the piping mechanism and, it is a trivially and consteval constructible/destructible standard layout, and its operator() is also constexpr (and satisfies consteval requirements).

The only formal requirement is that a noarr function is implemented as a callable object providing an `operator()` - during piping (`structure | function`), this operator receives the structure as its argument. The function then can provide a `func_family` typedef set to `top_tag`, `transform_tag`, or `get_tag` (see above, in piping).

- If `func_family` is set to `transform_tag`, the return value of `operator()` shall be a structure (otherwise it depends on the semantics of the function)

## High level abstractions and utilities

The library provides various high level abstractions and utilities that either simplify the usage of the lower level structures and functions, or they expand on them

### Wrapper

`wrapper` wraps a single structure and it provides the possibility of applying noarr functions (the expressions like `structure | function`) as methods using the traditional OO dot notation.

It can be created using the function `wrap`.

### Bag

`bag` combines a wrapped structure with an underlying data blob. It provides all `wrapper`'s methods that don't affect its layout (those which do affect the layout are generally those with `func_family` set to `transform_tag`).

It provides the following extra methods:

- `data()` returns the underlying data blob
- `structure()` returns the wrapped structure
- `at<Dims...>(values...)` returns a reference to a value in the data blob with the offset computed fixing dimensions named `Dims...` (`char` constants) to `values...` respectively. In other words it is a shortcut for `bag.structure() | get_at<Dims...>(bag.data(), values...)`

There are various types of `bag`s:

- vector bag

  - created by `make_vector_bag(structure)` or `make_vector_bag(wrapped_structure)`
  - the underlying data blob is implemented via a `std::vector`
  - the bag destroys the data blob in its destructor

- unique bag (the default type)

  - created by `make_unique_bag(structure)`, `make_unique_bag(wrapped_structure)`, `make_bag(structure)` or `make_bag(wrapped_structure)`
  - the underlying data blob is implemented via a `std::unique_ptr`
  - the bag destroys the data blob in its destructor

- observer bag

  - created by `make_bag(structure, char_carray)` or `make_bag(wrapped_structure, char_carray)`
  - the underlying data blob is implemented via `char *`
  - destruction of bag does not affect the data blob

### Structure printing

`print_struct(std::out&, structure)` prints the serialized type of the given structure to the provided output stream.

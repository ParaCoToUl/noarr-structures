# Dimension Kinds

A [structure](Glossary.md#structure) can have multiple [dimensions](Glossary.md#dimension).
Each dimension can be one of four kinds, depending on how the dimension was created.
There are slight differences in how these kinds can be used.

In general, noarr supports two ways to represent indices/lengths/offsets/sizes:
- **dynamic** values are only known at runtime (they may depend e.g. on program arguments or input).
  They are always represented using the `std::size_t` type.
  They occupy the necessary amount of bytes (usually 8) and cannot be derived from the type.
  Note that numeric literals are also considered dynamic unless marked otherwise (this is a property of C++ and cannot be overriden by noarr).
- **static** values are already known at compile time (but may be used at runtime).
  They are represented using the `std::integral_constant<std::size_t, *>` family of types.
  They usually take no space at all, since all the necessary information is stored in its type.
  Noarr provides a shortcut to create such values: `noarr::lit<N>`, e.g. `noarr::lit<42>`, without parentheses.

It is possible to add `using noarr::lit;` to the beginning of the file and then continue to only write e.g. `lit<42>`.
Static values can be used almost everywhere dynamic values can, since they have an implicit conversion to `std::size_t`.

Static values can *not* be used as template parameters or constexpr values (although GCC and some versions of Clang may allow this).
The proper way to extract the value is by taking the type of the expression and reading its `static constexpr std::size_t value` member:

```cpp
auto answer = lit<42>;

std::size_t a0 = answer; // correct, implicit conversion to dynamic, cannot ever be converted back or used as template parameter
std::size_t a1 = decltype(answer)::value; // correct, but unnecessarily complex

constexpr std::size_t a2 = decltype(answer)::value; // correct, can be used for template parameter
auto answer2 = lit<a2>; // correct, conversion back to integral_constant

constexpr std::size_t a3 = answer; // incorrect!
constexpr std::size_t a4 = answer.value; // incorrect!
```


## Vector-like dimensions (dynamic length, dynamic index)

In vector-like dimensions, the length is part of the structure, but it is not known at compile-time.
This is necessary for example when the number of elements depends on the program input.
Any value can be used as an index for this dimension, be it a static compile-time constant or a dynamically computed value, assuming it is in the range.


## Array-like dimensions (dynamic index, static length)

In array-like dimensions, the length is part of the structure type, known at compile-time, unlike the previous kind.
Any value can be still used as an index.


## Tuple-like dimensions (static length, static index)

The length is inherent in the type of the tuple structure -- it is the number of the tuple's structure template parameters.
Only static values can be used for indexing and the result type may depend on the index value.


## Unknown length dimensions

The length is not known and the structure itself has no way of computing it. It to be set from outside.

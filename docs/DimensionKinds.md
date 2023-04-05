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
  They are represented using the `std::integral_constant<std::size_t, N>` family of types.
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

constexpr std::size_t a3 = answer; // incorrect! (conversion from non-constexpr value)
constexpr std::size_t a4 = answer.value; // incorrect! (member of non-constexpr value, albeit static)
```


## Vector-like dimensions (dynamic length, dynamic index)

In vector-like dimensions, the length is part of the structure, but it is not known at compile-time.
This is necessary for example when the number of elements depends on the program input.
Any value can be used as an index for this dimension, be it a static compile-time constant or a dynamically computed value, assuming it is in the range.

The simplest way to create a structure with such a dimension is [`noarr::sized_vector`](structs/sized_vector.md):

```cpp
auto structure = noarr::scalar<float>() ^ noarr::sized_vector<'x'>(42);

auto size = structure | noarr::get_size(); // returns 42*sizeof(float) as a dynamic value
auto length = structure | noarr::get_length<'x'>(); // returns 42 as a dynamic value

auto doffset6 = structure | noarr::offset<'x'>(6); // returns 6*sizeof(float) as a dynamic value
auto soffset6 = structure | noarr::offset<'x'>(lit<6>); // returns 6*sizeof(float) as a static value
```

Note that in some cases (not this one), the length may depend on other parameters, some of which may not yet be known.
See [`noarr::into_blocks`](structs/into_blocks.md) for an example.

In the [signature](Signature.md), vector-like dimensions are represented as `noarr::function_sig<Dim, noarr::dynamic_arg_length, T>`.
[Traverser](Traverser.md) iterates vector-like dimensions using for-loops.
A single instance of the lambda body is created and it runs multiple times.


## Array-like dimensions (dynamic index, static length)

In array-like dimensions, the length is part of the structure type, known at compile-time, unlike the previous kind.
Any value can be still used as an index.

The previous example can be updated to an array-like by adding `lit<...>`:

```cpp
auto structure = noarr::scalar<float>() ^ noarr::sized_vector<'x'>(lit<42>); // <- added lit here

// different from vector-like
auto size = structure | noarr::get_size(); // returns 42*sizeof(float) as a *static* value
auto length = structure | noarr::get_length<'x'>(); // returns 42 as a *static* value

// same as vector-like
auto doffset6 = structure | noarr::offset<'x'>(6); // returns 6*sizeof(float) as a dynamic value
auto soffset6 = structure | noarr::offset<'x'>(lit<6>); // returns 6*sizeof(float) as a static value
```

Another way to create an array-like dimension is, as the name suggests, [`noarr::array`](structs/array.md):

```cpp
auto structure = noarr::scalar<float>() ^ noarr::array<'x', 42>();

// in this case, array will behave exactly the same as sized_vector(lit)
```

In the [signature](Signature.md), array-like dimensions are represented as `noarr::function_sig<Dim, noarr::static_arg_length<N>, T>`.
[Traverser](Traverser.md) iterates array-like in the same way as vector-like dimensions -- using for loops.


## Tuple-like dimensions (static length, static index)

The length is inherent in the type of the tuple structure -- it is the number of the tuple's structure template parameters.
Only static values can be used for indexing and the result type may depend on the index value.

The simplest way to create a structure with such a dimension is [`noarr::tuple`](structs/tuple.md):

```cpp
auto structure = noarr::make_tuple<'x'>(noarr::scalar<long>(), noarr::scalar<short>());

auto size = structure | noarr::get_size(); // returns sizeof(long)+sizeof(short) as a *static* value
auto length = structure | noarr::get_length<'x'>(); // returns 2 as a static value

auto soffset0 = structure | noarr::offset<'x'>(lit<0>); // returns 0 as a static value
auto soffset1 = structure | noarr::offset<'x'>(lit<1>); // returns sizeof(long) as a static value
auto soffset2 = structure | noarr::offset<'x'>(lit<2>); // fails at compile time (tuple index out of range)
auto doffset1 = structure | noarr::offset<'x'>(1); // fails at compile time (tuple index must be static)
```

In the [signature](Signature.md), tuple-like dimensions are represented as `noarr::dep_function_sig<Dim, T...>`.
[Traverser](Traverser.md) iterates tuple-like dimensions using template specialization.
The lambda body is specialized for each possible index.


## Unknown length dimensions

The length is not known and the structure itself has no way of computing it. It to be set from outside.

The simplest way to create a structure with such a dimension is [`noarr::vector`](structs/vector.md):

```cpp
auto structure = noarr::scalar<float>() ^ noarr::vector<'x'>();

// structure cannot be queried until the length is known
auto size = structure | noarr::get_size(); // fails at compile time (unknown vector length)

// option 0: set length in the structure, dynamically
auto structure42 = structure ^ noarr::set_length<'x'>(42);
auto size = structure42 | noarr::get_size(); // returns 42*sizeof(float) as a dynamic value

// option 1: set length in the structure, statically
auto structure42 = structure ^ noarr::set_length<'x'>(lit<42>);
auto size = structure42 | noarr::get_size(); // returns 42*sizeof(float) as a static value

// option 2: set length in the structure during the query, dynamically
auto state42 = noarr::make_state<noarr::length_in<'x'>>(42);
auto size = structure | noarr::get_size(state42); // returns 42*sizeof(float) as a dynamic value

// option 3: set length in the structure during the query, statically
auto state42 = noarr::make_state<noarr::length_in<'x'>>(lit<42>);
auto size = structure | noarr::get_size(state42); // returns 42*sizeof(float) as a static value

// ...
```

Note that `noarr::sized_vector<Dim>(len)` used in the first two kinds is actually just a shortcut for `noarr::vector<Dim>() ^ noarr::set_length<Dim>(len)`.

In the [signature](Signature.md), unknown length dimensions are represented as `noarr::function_sig<Dim, noarr::unknown_arg_length, T>`.
[Traverser](Traverser.md) cannot be used until the length is set.

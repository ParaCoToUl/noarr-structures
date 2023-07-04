# Conventions

This document describes some style and organizational conventions that might not be obvious
(it should not be necessary to document the formatting like indentation and bracing).


## Keywords

We use `struct`, `class` and `typename` in such a way that each has exactly one meaning:

- `struct` is a composite type declaration/definition (`class` is *not* used in this context)
- `class` is a type template parameter (`typename` is *not* used in this context)
- `typename` is a disambiguation for dependent type names (no equivalent choice here)

Example:

```cpp
struct foo { // not class
	using some_type = float; // not typedef
};

template<class Param> // not typename
struct generic_foo { // not class
	using some_type = typename Param::value_type;
};
```


## Types

Every type should satisfy the following (unless there is a reason not to):

- it should be a [standard layout type](https://en.cppreference.com/w/cpp/named_req/StandardLayoutType)
- the following should be trivial:
  - copy constructor
  - move constructor
  - destructor
- the following should be trivial or deleted:
  - default constructor
  - copy assignment
  - move assignment
- the default constructor should be present and trivial on empty types, and deleted on non-empty types

Additionally, when defining a nestable type, one should consider using [`noarr::contain`](Contain.md) as the base class instead of defining any fields.


## Functions

All functions should be `constexpr` unless prohibited by the standard.

All functions should be `noexcept`.
The standard allows a `noexcept` function to throw an exception (leading to `std::terminate`).
Hence, we can and do use `noexcept` even in functions that could throw an exception.

All member functions that do not modify `*this` should be `const`.


## Other

Returning new values is preferred to mutating existing objects.
Dynamic allocation and RAII is not used unless necessary ([`noarr::bag`](../BasicUsage.md#bag) and [TBB integration](../Traverser.md#traverser-range-and-tbb-integration))
to work with existing interfaces and conventions.
Types and functions that do so should be concentrated in the [`interop` source directory](../../include/noarr/structures/interop/).

The codebase should conform to the intersection of C++20 and C++23.

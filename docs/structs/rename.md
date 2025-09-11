# rename

Rename the specified [dimensions](../Glossary.md#dimension) in a structure.

```hpp
#include <noarr/structures_extended.hpp>

template<typename T, auto... DimPairs>
struct noarr::rename_t;

template<auto... DimPairs>
constexpr proto noarr::rename();
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

The structure `rename_t` has the same layout and properties as the `T` wrapped in it, except that some dimensions have different names.
The old names are removed from the structure. The `rename` function creates a proto-structure for `rename_t`.

The dimension names come in pairs (as opposed to, e.g. all old names, then all new names).
For example, `noarr::rename<'a', 'b', 'c', 'd'>()` renames `'a'` to `'b'` and `'c'` to `'d'`.
It is allowed to swap (or otherwise permute) the dimension names.


## Usage examples

Dimension renaming can be used to make two structures compatible (for traversal or indexing with the same state).
In the following example, we copy a vector indexed by `'i'` into one indexed by `'x'`:

```cpp
// let's say we cannot change these definitions
auto from = noarr::make_bag(noarr::scalar<float>() ^ noarr::vector<'i'>(42), /*...*/);
auto to = noarr::make_bag(noarr::scalar<float>() ^ noarr::vector<'x'>(42));

// create a view of `from` that is compatible with `to`
auto from_view = from ^ noarr::rename<'i', 'x'>();

// copy as if they had the same structure
noarr::traverser(to).for_each([&](auto state) {
	to[state] = from_view[state];
});
```

A similar use-case is when an algorithm prescribes the dimension a structure must have:

```cpp
template<typename ABag, typename BBag, typename CBag>
void matmul(ABag a, BBag b, CBag c) {
	// a has i, j
	// b has j, k
	// c has i, k
	// ...
}

auto matrix_struct = noarr::scalar<float>() ^ noarr::array<'i', 3>() ^ noarr::array<'j', 3>();
auto a = noarr::make_bag(matrix_struct, /*...*/);
auto b = noarr::make_bag(matrix_struct, /*...*/);
auto c = noarr::make_bag(matrix_struct, /*...*/);

matmul(
	a,
	b ^ noarr::rename<'i', 'j', 'j', 'k'>(),
	c ^ noarr::rename<'j', 'k'>()
);
```

Last, renaming the dimensions can be used to change the interpretation of a structure:

```cpp
auto a = noarr::make_bag(noarr::scalar<float>() ^ noarr::array<'i', 3>() ^ noarr::array<'j', 3>(), /*...*/);
auto a_transposed_view = a ^ noarr::rename<'i', 'j', 'j', 'i'>();
```

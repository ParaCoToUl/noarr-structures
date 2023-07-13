# Mangling

Render the full description of a [structure](../Glossary.md#structure) to a piece of C++ source code.
It is possible to mangle either structure types or structure objects (i.e. just metadata, not elements, for that, see [Serialization](Serialization.md)).

```hpp
#include <noarr/structures/extra/mangle.hpp>

template<char... Chars>
using noarr::char_sequence = std::integer_sequence<char, Chars...>;

template<typename Struct>
using noarr::mangle = noarr::char_sequence</*...*/>;

template<typename Struct>
struct noarr::mangle_to_str {
	static constexpr char c_str[] = {/*...*/, '\0'};
	static constexpr std::size_t length = /*...*/;
};

template<typename String>
constexpr String noarr::mangle_expr(const auto &structure);
```

The `mangle` and `mangle_to_str` templates mangle just the type of the structure. `mangle` is only useful for further processing with templates.
To obtain a string, use `mangle_to_str`, which converts to a pair of static constexpr variables.
The following example shows how these can be treated as a sized (C++ style) string or a null-terminated (C style) string:

```cpp
using struct_type = noarr::scalar<float>;

using s = noarr::mangle_to_str<struct_type>;
// use as sized
assert(std::memcmp(s::c_str, "scalar<float>", s::length) == 0);
// use as null-terminated
assert(std::strcmp(s::c_str, "scalar<float>") == 0);
// convert to std::string (two options)
assert(std::string(s::c_str, s::length) == "scalar<float>");
assert(std::string(s::c_str) == "scalar<float>");

// for templates
static_assert(std::is_same_v<noarr::mangle<struct_type>, std::integer_sequence<char, 's', 'c', 'a', 'l', 'a', 'r', '<', 'f', 'l', 'o', 'a', 't', '>'>>);
```

The `mangle_expr` function mangles both the type and the metadata (e.g. lengths, block sizes, etc).
It can be used with `std::string` or anything default-constructible that has `.push_back(char)` and `.append(char*, std::size_t)`.
The following example uses `std::string`:

```cpp
auto struct_full = noarr::scalar<float>();

assert(noarr::mangle_expr<std::string>(struct_full) == "scalar<float>{}");
```

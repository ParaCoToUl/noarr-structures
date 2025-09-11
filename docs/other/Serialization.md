# Serialization

Read the [structure](../Glossary.md#structure) elements from a text stream or write them to a text stream.

```hpp
#include <noarr/structures/interop/serialize_data.hpp>

constexpr decltype(auto) noarr::deserialize_data(auto &&in, auto structure, void *data);
constexpr decltype(auto) noarr::deserialize_data(auto &&in, auto &&bag);
constexpr decltype(auto) noarr::serialize_data(auto &&out, auto structure, const void *data);
constexpr decltype(auto) noarr::serialize_data(auto &&out, const auto &bag);
```

In the above signatures, `in` is a [`std::basic_istream`](https://en.cppreference.com/w/cpp/io/basic_istream),
`out` is a [`std::basic_ostream`](https://en.cppreference.com/w/cpp/io/basic_ostream),
and `bag` is a [`noarr::bag`](../BasicUsage.md#bag). The passed stream is returned (using forwarding).

The output format is line based, one number per line, with one trailing newline at the end.
The input format is similar, but any number of any whitespace is accepted as the separator.
No additional data after the last element are consumed.
It is possible to write/read multiple structures to the same stream if the order and sizes are preserved.

No metadata are written or read. The structure must be allocated before reading.
The number format depends on the stream formatting state.
Error and EOF conditions can be checked on the stream using the standard state functions.

Examples:

```cpp
auto bag = noarr::make_bag(/*...*/);

if(!noarr::deserialize_data(std::ifstream("path/to/src"), bag)) {
	std::cerr << "Input error" << std::endl;
	return 1;
}

// ... <- Do some transformation on the data

if(!noarr::serialize_data(std::ofstream("path/to/dest"), bag)) {
	std::cerr << "Output error" << std::endl;
	return 1;
}
```

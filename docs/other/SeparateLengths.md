# Separate Lengths

The [`set_length`](../structs/set_length.md) proto-structure is not the only way to specify a structure length.
It is possible to pass it in a [state](../State.md):

```cpp
// A structure with no lengths
auto my_matrix = noarr::scalar<float>() ^ noarr::vector<'i'>() ^ noarr::vector<'j'>();

auto state = noarr::idx<'i', 'j'>(2, 1).template with<noarr::length_in<'i'>, noarr::length_in<'j'>>(3, 3);
std::size_t thirty_six = my_matrix | noarr::get_size(state);
std::size_t twenty = my_matrix | noarr::offset(state);
// ...
```

This is in fact what `set_length<Dim>` does under the hood: it adds a `noarr::length_in<Dim>` to the state and passes it on to its [sub-structure](../Glossary.md#sub-structure).

The definition of `state` above could be equivalently rewritten as follows (`noarr::idx` and `with` are just shortcuts):

```cpp
auto state = noarr::make_state<noarr::index_in<'i'>, noarr::index_in<'j'>, noarr::length_in<'i'>, noarr::length_in<'j'>>(2, 1, 3, 3);
```


## In traverser

Normally, [traverser](../Traverser.md) can only work with structures of known lengths.
However, it is enough to set the length in the traverser [`.order()`](../Traverser.md#orderproto-structure-customizing-the-traversal).
It will then also be available in the state passed to the lambda:

```cpp
noarr::traverser(my_matrix).order(noarr::set_length<'i', 'j'>(3, 3)).for_each([my_matrix](auto state) {
	// These will always pass, since set_length puts the lengths in the state
	assert(state.template get<noarr::length_in<'i'>>() == 3);
	assert(state.template get<noarr::length_in<'j'>>() == 3);

	// These will not even compile, because the length is not set in my_matrix
	auto i_len = my_matrix | noarr::get_length<'i'>();
	auto j_len = my_matrix | noarr::get_length<'j'>();

	// This will work thanks to the state
	std::size_t off = my_matrix | noarr::offset(state); // Or get_at
	// ...
});
```

# Iterator

The `noarr::iterate` function returns an object that can be used with a [range-based for loop](https://en.cppreference.com/w/cpp/language/range-for).
Its operation is similar to the [traverser](../Traverser.md) `for_dims` method,
in that it receives a [structure](../Glossary.md#structure) and a list of [dimension](../Glossary.md#dimension) names and iterates the structure along these dimensions.

While the syntax is more convenient, an iterator like this cannot be implemented efficiently in C++.
As such, you shouldn't use `noarr::iterate` and its iterators in any calculations where high performance and/or vectorization is expected.
It is fine to use it in code that handles text I/O, or only ever works with small structures, or similar.

Note that `noarr::iterate` is different from [using a *traverser* with the range-based for loop](../Traverser.md#traverser-iterator):
the latter only iterates one dimension of the structure (and is efficient, unlike `noarr::iterate`).

The value type of the iterator is a [state](../State.md).

```cpp
auto structure = noarr::scalar<double>() ^ noarr::sized_vector<'x'>(20) ^ noarr::sized_vector<'y'>(30);

for(auto state : structure | noarr::iterate<'y', 'x'>()) {
	auto x = noarr::get_index<'x'>(state);
	auto y = noarr::get_index<'y'>(state);
	auto off = structure | noarr::offset(state);
	// ...
}
```

As any [other noarr function](../BasicUsage.md#functions), `noarr::iterate` can also be used with a [bag](../BasicUsage.md#bag):

```cpp
auto bag = noarr::make_bag(structure);

for(auto state : bag | noarr::iterate<'y', 'x'>()) {
	// same options as in the previous example, but now we also have the bag
	bag[state] = drand48();
}
```

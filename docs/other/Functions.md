# Other Functions

In addition to [get_length], [get_size], [offset], and [get_at], the library provides the following functions.


## offset_of

`offset_of<SubStructType>` is a generalization of [`offset`](../BasicUsage.md#offset), which returns the offset of some [sub-structure](../Glossary.md#sub-structure) other than the scalar.
There is no similar generalization of [`get_at`](../BasicUsage.md#get_at), because it could not return a reference (there may be no C++ type corresponding to non-scalar structures).


## iterate

`noarr::iterate<Dims...>()`, which constructs an [iterator](Iterator.md) is also a function.


## Custom

It is relatively easy to define a custom function.
The `|` operator accepts anything that can be called with a single argument (which is a [structure](../Glossary.md#structure)).
Note that to accept any structure type, it must be generic. However, a function template name cannot be used without calling it:

```cpp
template<typename Structure>
auto get_width(Structure s) {
	return s | noarr::get_length<'x'>();
}

auto w = matrix | get_width; // this will not work (using template without call)
auto w = matrix | get_width(); // this does not make sense (the function expects one argument)
auto w = get_width(matrix); // works but not what we wanted
```

The solution is to use an object with a generic `operator()`. For example a lambda:

```cpp
constexpr auto get_width = [](auto s) {
	return s | noarr::get_length<'x'>();
};

auto w = matrix | get_width;
```

Note that there are no parentheses, since we do not call the function (the `|` operator does).
If you want the function to take arguments (or want the empty parentheses), create a function that returns a lambda:

```cpp
//template<...>?
constexpr auto get_width(/*args?*/) {
	return [](auto s) {
		return s | noarr::get_length<'x'>();
	};
}

auto w = matrix | get_width();
```

Alternatively, you can define a type (the call-like expression will then be a default-initialization).

```cpp
struct get_width {
	// add custom constructor to have args

	template<typename Structure>
	auto operator()(Structure s) {
		return s | noarr::get_length<'x'>();
	}
};

auto w = matrix | get_width(); // same as get_width{}
```

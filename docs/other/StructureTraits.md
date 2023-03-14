# Structure Traits

Structure traits (named after the standard header `<type_traits>`)
are templates to check basic properties of [structures](../Glossary.md#structure) and [signatures](../Glossary.md#signature).

```hpp
#include <noarr/structures/extra/struct_traits.hpp>

template<typename Signature>
struct noarr::sig_is_point; // : std::false_type or std::true_type

template<typename Struct>
struct noarr::is_point; // : std::false_type or std::true_type

template<typename Signature>
struct noarr::sig_is_cube; // : std::false_type or std::true_type

template<typename Struct>
struct noarr::is_cube; // : std::false_type or std::true_type

template<typename Signature, typename State>
struct noarr::sig_get_scalar {
	using type = /*...*/;
};

template<typename Struct, typename State = noarr::state<>>
using noarr::scalar_t = /*...*/;
```

The three templates with a `sig_` prefix work with signatures, the other three work with structure types.

In the above names, `point` refers to a structure that has no [dimensions](../Glossary.md#dimension).
This is true for a [scalar](../structs/scalar.md), but also for any structure which has all its dimensions [fixed](../structs/fix.md).

`cube` is a structure that allows all indices to be dynamic (i.e. it has no tuple-like dimensions, see [Dimension Kinds](../DimensionKinds.md)).
In other words, it is a (multidimensional) matrix. Any `point` is a `cube`, as are most other structures.
For a structure to *not* be a cube, it must have an *unfixed* [tuple](../structs/tuple.md) dimension (or similar).
The main advantage of a cube in comparison to a non-cube is that all its values are of the same type (see the next paragraph).

`sig_get_scalar` and `scalar_t` return the element type of a structure.
If there are any tuple-like dimensions (i.e. the structure is not a `cube` as defined above), a non-empty [state](../State.md) is needed.
Specifically, the state must contain a static index for any tuple-like dimension.
Note that `::type` must be used on `sig_get_scalar`, while `scalar_t` returns the type directly.

All these functions are simple utilities implemented by inspecting the signature.
When this is not enough, one can [use the signature directly](../Signature.md).

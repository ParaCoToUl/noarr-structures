# vector

Repeat the [sub-structure](../Glossary.md#sub-structure) layout, forming a vector of elements matching the sub-structure,
and use a new [dimension](../Glossary.md#dimension) (of an unknown length) to select between the elements.

```hpp
#include <noarr/structures.hpp>

template<auto Dim, typename T>
struct noarr::vector;

template<auto Dim>
constexpr proto noarr::vector();
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

`vector` is the main way to define custom noarr structures and their dimensions.

The layout of a `vector` structure consists of several copies of the layout of `T`, one after another without any padding.
The whole structure has one dimension more than `T`, the new dimension has name `Dim`.
The [index](../Glossary.md#index) in `Dim` is used to select one of the copies of the `T` layout.

Neither the `vector` structure itself nor the protostructure created by `vector()` set the length of the new dimension - it must be [set externally](../BasicUsage.md#lengths) by applying `noarr::set_length` via the `^` operator. This can be shortened by using `noarr::vector<Dim>(length)`.

The size of a vector is the sub-structure size multiplied by the vector length. The vector does not add any padding.

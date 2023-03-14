# scalar

A structure with no [dimensions](../Glossary.md#dimension), containing just one scalar value.

```hpp
#include <noarr/structures.hpp>

template<typename T>
struct noarr::scalar;
```


## Description

The simplest structure. It has exactly one element (of type `T`, which can be e.g. a primitive type), no [sub-structures](../Glossary.md#sub-structure).
It serves as the leaf substructure in structure hierarchies and as the bottom case for many mechanisms defined by the library.
Its size matches the size of the contained type and it is always known during compile time.

# Contain

`noarr::strict_contain` facilitates the creation of new structures. It is a tuple-like struct that defines a struct's fields via inheritance,
but in contrast with [`std::tuple`](https://en.cppreference.com/w/cpp/utility/tuple), it is a trivially constructible standard layout.


## Advantages

The main advantage of `noarr::strict_contain` is that it does not actually store or allocate space for empty objects,
thus avoiding a shortcoming of C++, which requires empty objects to have size 1 (not 0).
Many objects in noarr are empty and what matters is their type (not address), so storing them in `noarr::strict_contain` may be useful in some cases.


## Use

Most noarr types (especially structures) use `noarr::strict_contain` as the base type instead of defining any fields.
In most client code the advantages of using it would be vanishing. We recommend to use it only when [defining custom structures](../DefiningStructures.md#defining-structures-manually).


## Limitations

Unlike the general `std::tuple`, `noarr::strict_contain` should only be used to store [simple enough](Conventions.md#types) types.

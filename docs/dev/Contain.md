# Contain

`noarr::contain` facilitates the creation of new structures. It is a tuple-like struct that defines a struct's fields via inheritance,
but in contrast with [`std::tuple`](https://en.cppreference.com/w/cpp/utility/tuple), it is a trivially constructible standard layout.


## Advantages

The main advantage of `noarr::contain` is that it does not actually store or allocate space for empty objects,
thus avoiding a shortcoming of C++, which requires empty objects to have size 1 (not 0).
Many objects in noarr are empty and what matters is their type (not address), so storing them in `noarr::contain` may be useful in some cases.


## Use

Most noarr types (especially structures) use `noarr::contain` as the base type instead of defining any fields.

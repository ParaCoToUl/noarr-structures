# Signature

**Note:** Signatures are a relatively low-level feature and should not be needed unless you are:
- trying to inspect structure types passed to your code as template arguments
- implementing your own structure type [by hand](DefiningStructures.md#defining-structures-manually)
- curious.


## Introduction

The signature of a [structure](Glossary.md#structure) describes the interface of the structure.
Signatures are purely compile-time objects (represented by C++ types without values) and can be obtained from the `::signature` member typedef of structure types.

For the purpose of this document, we will view structures as function types.
For example, an array of floats is a function that takes one integer parameter (the index) and returns float.
Written in a functional syntax: `size_t -> float`.
A two-dimensional array is a higher-order function: after we pass the first index, we get a one-dimensional array.
In the same syntax, a 2D float array will be `size_t -> size_t -> float`.

In noarr, however, the order of dimensions is not as important as their names. The signature will have to take this into account.
Therefore, a more precise expression would be `(x: size_t) -> (y: size_t) -> float` (assuming the array's dimensions are named `'x'` and `'y'`).

Last but not least, not all dimensions in noarr are created equal.
See [Dimension Kinds](DimensionKinds.md) for details about how different dimensions can and cannot be used -- reading that document should *not* be necessary to understand signatures.


## Signature classes

There are three signature types. The intended usage is pattern matching using template specialization on concrete types. Adding custom signature types is thus not allowed.
The definition of a signature is recursive. We will start with the "base case":

### Scalar

A signature in the form of `noarr::scalar_sig<T>` (where `T` is any C++ object type) means that the structure does not accept or expect any indices in any dimensions.
An example of such a structure is [`noarr::scalar<T>`](structs/scalar.md), although it is not the only example.

### Function

This kind of structure is the most common. It also includes the array example from the [introduction](#introduction).

The signature of such structures is in the form `noarr::function_sig<Dim, ArgLength, RetSig>` where
- `Dim` is a dimension name (`char`)
- `ArgLength` is one of `noarr::unknown_arg_length`, `noarr::dynamic_arg_length`, `noarr::static_arg_length<...>` (see below)
- `RetSig` is another signature type

A structure that has a signature like this accepts an index in dimension `Dim` and otherwise behaves the same as a structure with signature `RetSig`.
This is the recursive step: if `RetSig` is N-dimensional signature (i.e. it takes N arguments), then the whole `noarr::function_sig<Dim, ArgLength, RetSig>` is (N+1)-dimensional.

The `ArgLength` further specifies the [length](Glossary.md#length) in dimension `Dim`.
- `noarr::unknown_arg_length` means the length in the dimension has not been set or computed.
  When it is necessary for size and index calculation, you will need to either pass `length_in<Dim>` in the [state](State.md) or [set the length](structs/set_length.md) in the structure.
- `noarr::dynamic_arg_length` means the length is either set to a value that is not a compile-time constant, or it is calculated dynamically from some other dimension.
  In the latter case, it may be necessary to set some other parameters before the length in this dimension may be queried.
  In either case, the length cannot be set or passed from outside: any attempt to do so (e.g. `length_in<Dim>` in the state) will be ignored.
- `noarr::static_arg_length<...>` (for example `noarr::static_arg_length<8192>`) means the length has been set to a compile-time constant (of type `size_t`).
  Like with `noarr::dynamic_arg_length`, the length cannot be set or passed from outside and such attempts are ignored.

### Dependent function

This type of signature is currently only used for tuples. It is represented as `noarr::function_sig<Dim, RetSigs...>`.
It is similar to the Function type: it accepts an index in `Dim` in addition to the indices accepted by `RetSigs`.

The difference is that where a Function returns the same type for any value of index, a Dependent function may return a different type for each possible value of index.
In other words: what other dimensions are accepted by the structure depends on the element of `RetSigs` that is choosen according to the index in `Dim`.
(This places a further requirement on the type of index that can be passed, see [Dimension Kinds](DimensionKinds.md) for more details.)

Also, unlike Function, there is no need for an `ArgLength` parameter: the [length](Glossary.md#length) in `Dim` always corresponds to the count of `RetSigs` variadic parameters.


## Working with signatures

Before using a signature directly, consider using one of the [structure traits](other/StructureTraits.md) utilities, which may be easier, shorter, and more readable.

That being said, direct usage of signatures is also possible and supported. As mentioned above, the intended usage is C++ template specialization:

```cpp
// The primary template may be left undefined, since all legal signatures are handled by the three specializations.
// Using an illegal signature (i.e. any other type than the three described above) result in compile-time error.
template<class T>
struct foo;

template<char Dim, class ArgLength, class RetSig>
struct foo<noarr::function_sig<Dim, ArgLength, RetSig>> {
	// Inspect Dim, ArgLength, and RetSig:
	static constexpr bool xxx = ArgLength::is_known; // true for false for dynamic_arg_length and static_arg_length, false for unknown_arg_length
	static constexpr bool yyy = ArgLength::is_static; // true for false for static_arg_length, false for dynamic_arg_length and unknown_arg_length
	static constexpr bool zzz = ArgLength::value; // beware! only valid if ::is_static, i.e. if ArgLength is static_arg_length<N> (::value is the N)
	using example_recursion = foo<RetSig>;
};

template<char Dim, class... RetSigs>
struct foo<noarr::dep_function_sig<Dim, RetSigs...>> {
	// Inspect Dim and RetSigs:
	static constexpr bool zzz = sizeof...(RetSigs);
	// Alternatively, reconstruct dep_function_sig<Dim, RetSigs...> and use its static members:
	using orig_sig = noarr::dep_function_sig<Dim, RetSigs...>;
	using elem_3_sig = typename orig_sig::template ret_sig<3>; // extract the third element - this is an example, there will usually be some computation instead of just 3
};

template<class ValueType>
struct foo<noarr::scalar_sig<ValueType>> {
	// Use ValueType. It is the just the type without any additional wrappers. E.g. float, int, ...
};
```

For quick access to some oft-used properties, there are also some built-in utilities in the signature types.
- `::dependent` is a `static constexpr bool`. `true` for `dep_function_sig`, `false` for `function_sig`, absent for `scalar_sig`.
- `::template any_accept<Dim>` and `::template all_accept<Dim>` are `static constexpr bool`s. They detect whether the signature contains a `Dim` at any level.
  These two differ in the treatment of Dependent function signatures (tuples):
  `all_accept` requires that there is a `Dim` in every component, for `any_accept` just one such component is enough.
- `::template replace<Replacement, Dim>` is a typedef (`Replacement` must be a type template that takes a signature and results in a signature).
  It finds an occurrence of `Dim` (either `function_sig<Dim, *>` or `dep_function_sig<Dim, *>`) and replaces the whole matching signature (`S`) with `Replacement<S>::type`.
  In case of tuples, the replacement is done in each component.

Note that signatures that are specialized to unexpected types (e.g. non-signature `RetSig` or `ArgLength` that is not one of the three enumerated) are invalid.
Also, a signature that contains two nested Functions or Dependent functions with the same dimension name is invalid
(having the same name in multiple components of a tuple is OK, provided there is no nesting).
Such signatures are never created by noarr, and user code must take care not create them too. None of the built-in utilities detect invalid signatures.

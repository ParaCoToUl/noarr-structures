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
Therefore, a more accurate expression would be `(x: size_t) -> (y: size_t) -> float` (assuming the array's dimensions are named `'x'` and `'y'`).

Last but not least, not all dimensions in noarr are created equal.
See [Dimension Kinds](DimensionKinds.md) for details about how different dimensions can and cannot be used -- reading that document should *not* be necessary to understand signatures.


## Signature classes

There are three signature types. The intended usage is pattern matching using template specialization on concrete types. Adding custom signature types is thus not allowed.
The definition of a signature is recursive. We will start with the "base case":

### Scalar

A signature in the form of `noarr::scalar_sig<T>` (where `T` is any C++ object type) means that the structure does not accept or expect any indices in any dimensions.
An example of such a structure is [`noarr::scalar<T>`](structs/scalar.md), although it is not the only example (see [signature examples](#examples)).

### Function

This kind of structure is the most common. It also includes the array example from the [introduction](#introduction), and [many others](#examples).

The signature of such structures is in the form `noarr::function_sig<Dim, ArgLength, RetSig>` where
- `Dim` is a dimension name (`auto`)
- `ArgLength` is `noarr::dynamic_arg_length` or `noarr::static_arg_length<...>` (see below)
- `RetSig` is another signature type

A structure that has a signature like this accepts an index in dimension `Dim` and otherwise behaves the same as a structure with signature `RetSig`.
This is the recursive step: if `RetSig` is N-dimensional signature (i.e. it takes N arguments), then the whole `noarr::function_sig<Dim, ArgLength, RetSig>` is (N+1)-dimensional.

The `ArgLength` further specifies the [length](Glossary.md#length) in dimension `Dim`.
- `noarr::dynamic_arg_length` means the length is either set to a value that is not a compile-time constant, or it is calculated dynamically from some other dimension.
  In the latter case, it may be necessary to set some other parameters before the length in this dimension may be queried (see [`noarr::into_blocks`](structs/into_blocks.md) for an example).
  In either case, the length cannot be set or passed from outside: any attempt to do so (e.g. `length_in<Dim>` in the state) will be ignored.
- `noarr::static_arg_length<...>` (for example `noarr::static_arg_length<8192>`) means the length has been set to a compile-time constant (of type `size_t`).
  Like with `noarr::dynamic_arg_length`, the length cannot be set or passed from outside and such attempts are ignored.

### Dependent function

This type of signature is most prominently used for tuples. It is represented as `noarr::dep_function_sig<Dim, RetSigs...>`.
It is similar to the Function type: it accepts an index in `Dim` in addition to the indices accepted by `RetSigs`.

The main difference is that in the return type.
Where a Function returns the same type for any value of index, a Dependent function may return a different type for each possible value of index.
In other words: what other dimensions are accepted by the structure depends on the element of `RetSigs` that is chosen according to the index in `Dim`.
(This places a further requirement on the type of index that can be passed, see [tuple-like dimensions](DimensionKinds.md#tuple-like-dimensions-static-length-static-index) for more details.)

Also, unlike Function, there is no need for an `ArgLength` parameter: the [length](Glossary.md#length) in `Dim` always corresponds to the count of `RetSigs` variadic parameters.


## Working with signatures

Before using a signature directly, consider using one of the [structure traits](other/StructureTraits.md) utilities, which may be easier, shorter, and more readable.

That being said, direct usage of signatures is also possible and supported. As mentioned above, the intended usage is C++ template specialization:

```cpp
// The primary template may be left undefined, since all legal signatures are handled by the three specializations.
// Using an illegal signature (i.e. any other type than the three described above) result in compile-time error.
template<class T>
struct foo;

template<auto Dim, class ArgLength, class RetSig>
struct foo<noarr::function_sig<Dim, ArgLength, RetSig>> {
	// Inspect Dim, ArgLength, and RetSig:
	static constexpr bool xxx = ArgLength::is_known; // true for false for dynamic_arg_length and static_arg_length, false for dynamic_arg_length
	static constexpr bool yyy = ArgLength::is_static; // true for false for static_arg_length, false for dynamic_arg_length and dynamic_arg_length
	static constexpr bool zzz = ArgLength::value; // beware! only valid if ::is_static, i.e. if ArgLength is static_arg_length<N> (::value is the N)
	using example_recursion = foo<RetSig>;
};

template<auto Dim, class... RetSigs>
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
  See [the example of a replacer](#implementing-a-replacer). In case of tuples, the replacement is done in each component.

Note that signatures that are specialized to unexpected types (e.g. non-signature `RetSig` or `ArgLength` that is not one of the three enumerated) are invalid.
Also, a signature that contains two nested Functions or Dependent functions with the same dimension name is invalid
(having the same name in multiple components of a tuple is OK, provided there is no nesting).
Such signatures are never created by noarr, and user code must take care not create them too. None of the built-in utilities detect invalid signatures.


## Examples

### Adding dimensions

The signature of a structure is fixed. However, when building a more complex structure from simpler ones, the resulting signatures are often very similar, only with small modifications.
The simplest transformation is adding an index. When a structure is wrapped in an array or similar structure, the signature is also wrapped in an additional layer.

```cpp
auto elem = noarr::scalar<float>();
using elem_sig = noarr::scalar_sig<float>;
static_assert(std::is_same_v<decltype(elem)::signature, elem_sig>); // should pass

auto arr1D = elem ^ noarr::array<'x', 42>();
using arr1D_sig = noarr::function_sig<'x', noarr::static_arg_length<42>, elem_sig>;
static_assert(std::is_same_v<decltype(arr1D)::signature, arr1D_sig>); // should pass

auto arr2D = arr1D ^ noarr::array<'y', 54>();
using arr2D_sig = noarr::function_sig<'y', noarr::static_arg_length<54>, arr1D_sig>;
static_assert(std::is_same_v<decltype(arr2D)::signature, arr2D_sig>); // should pass
```

Just a variant of the above is [`noarr::vector`](structs/vector.md), which does not need to have a statically fixed length:

```cpp
// using elem from above
auto vec1D = elem ^ noarr::vector<'x'>();
using vec1D_sig = noarr::function_sig<'x', noarr::dynamic_arg_length, elem_sig>; // dynamic_arg_length instead of static_arg_length
```

### Modifying dimensions

So far, the signature has just reflected the organization of the structure. However, this is not the case for most structures.
The [`set_length`](structs/set_length.md) structure does not add any layers, but rather modifies existing layers.
In the following example, we set the length of a vector to a compile-time constant, making it the same as array.
Note that this will *not* result in an array structure type, but the structure *will* have the same signature as the array.

```cpp
// using elem from above
auto vec1D_sized = elem ^ noarr::vector<'x'>() ^ noarr::set_length<'x'>(noarr::lit<42>);
using vec1D_sized_sig = noarr::function_sig<'x', noarr::static_arg_length<42>, elem_sig>; // exactly the same as arr1D_sig
```

We used `noarr::lit` to make sure the expression is constant. Without it, we will end up with a dynamic size.

```cpp
// using elem from above
auto vec1D_dynsized = elem ^ noarr::vector<'x'>() ^ noarr::set_length<'x'>(42);
using vec1D_dynsized_sig = noarr::function_sig<'x', noarr::dynamic_arg_length, elem_sig>;
```

### Replacing dimensions

Most structures not only add and modify dimensions. For example, [`noarr::into_blocks`](structs/into_blocks.md) adds new dimensions, but also removes one:

```cpp
// same as previous, just rewritten as one-liners
auto arr2D = noarr::scalar<float>() ^ noarr::array<'x', 42>() ^ noarr::array<'y', 54>();
using arr2D_sig = noarr::function_sig<'y', noarr::static_arg_length<54>,
                   noarr::function_sig<'x', noarr::static_arg_length<42>,
                    noarr::scalar_sig<float> > >;

auto blocked = arr2D ^ noarr::into_blocks<'x', 'u', 'v'>(); // removes x, adds u and v
using blocked_sig = noarr::function_sig<'y', noarr::static_arg_length<54>,
                     noarr::function_sig<'u', noarr::dynamic_arg_length, // u is dynamic, its length is computed from the lengths of x/v
                      noarr::function_sig<'v', noarr::dynamic_arg_length, // v is unknown, its length must be set externally
                       noarr::scalar_sig<float> > > >;
```

Since the index for `'x'` is computed from the indices for `'u'` and `'v'`, it no longer makes sense to set `'x'` from outside.
In fact, the structure behaves as if it never had an `'x'` dimension, and the signature reflects this.
The signature should describe the interface of the structure, as opposed to its implementation.
For this reason, `'x'` will be completely absent from the signature.

### Removing dimensions

In the same way `into_blocks` replaces a dimension with two new ones, [`noarr::fix`](structs/fix.md) replaces it with nothing (zero new dimensions).

```cpp
// same as previous, just rewritten as one-liners
auto arr2D = noarr::scalar<float>() ^ noarr::array<'x', 42>() ^ noarr::array<'y', 54>();
using arr2D_sig = noarr::function_sig<'y', noarr::static_arg_length<54>,
                   noarr::function_sig<'x', noarr::static_arg_length<42>,
                    noarr::scalar_sig<float> > >;

auto fixed = arr2D ^ noarr::fix<'x'>(6); // removes x, adds nothing
using fixed_sig = noarr::function_sig<'y', noarr::static_arg_length<54>,
                   noarr::scalar_sig<float> >;
```

Again, this reflects the fact that `'x'` cannot be used from outside.

### Non-scalar structures with scalar signatures

An extreme example of the above is fixing all dimensions (here we have only one for simplicity):

```cpp
auto arr1D = noarr::scalar<float>() ^ noarr::array<'x', 42>() ^ noarr::fix<'x'>(6);
using arr1D_sig = noarr::scalar_sig<float>;
```

### Obtaining the signature of a parameter

Taking the signature of a structure that is defined one line above is not very useful.
Signatures are intended to allow generic algorithms and functions to work with any structure passed to them via template parameters.

Note that due to C++ limitations, `::signature` will not work when used on template parameters (or anything derived from them). You will need to add `typename`:

```cpp
// obtaining signature from type directly:
template<typename S>
void foo(S structure) {
	using sig = typename S::signature;
	// ...
}

// obtaining signature when no type name is available (e.g. in lambda):
auto lambda_foo = [](auto structure) {
	using sig = typename decltype(structure)::signature;
	// ...
};
```

### Implementing a replacer

Recall that every signature has a `::template replace<Replacement, Dim>` member that finds a piece of signature by dimension name and replaces it.
The following example renames a dimension `'x'` to `'y'` by replacing the appropriate piece of signature with a similar one with different name:

```cpp
// could be template parameters of some larger structure
static constexpr char old_dim = 'x', new_dim = 'y';

template<class Original>
struct replacement;

template<class ArgLength, class RetSig>
struct replacement<noarr::function_sig<old_dim, ArgLength, RetSig>> {
	// The "type" member is required by ::replace. We return the same structure, just with the name old_dim replaced by new_dim.
	// RetSig contains the rest of the structure - we need not to process that, so we just use it unchanged. Likewise with ArgLength.
	using type = noarr::function_sig<new_dim, ArgLength, RetSig>;
};

template<class... RetSigs>
struct replacement<noarr::dep_function_sig<old_dim, RetSigs...>> {
	// Analogic to the above.
	using type = noarr::dep_function_sig<old_dim, RetSigs...>;
};

using old_sig = /*...*/::signature;
using new_sig = typename old_sig::template replace<replacement, old_dim>;

#if WANT_DEAD_CODE
// Note that the following specializations are never needed.
// ::replace will never try to instantiate the Replacement template for something that does not match the requested dimension name.

template<auto Dim, class ArgLength, class RetSig>
struct replacement<noarr::function_sig<Dim, ArgLength, RetSig>> { /*...*/ }; // dimension name Dim other than the requested old_dim
template<auto Dim, class... RetSigs>
struct replacement<noarr::dep_function_sig<Dim, RetSigs...>> { /*...*/ }; // dimension name Dim other than the requested old_dim
template<class ValueType>
struct replacement<noarr::scalar_sig<ValueType>> { /*...*/ }; // no dimension name at all

#endif
```


## Relation to State

Among other, a signature describes the items that the structure expects to receive in the [state](State.md) and their types.

Roughly speaking, the state should contain an `noarr::index_in<Dim>` for each dimension mentioned in the signature.
It should also contain a `noarr::length_in<Dim>` for each dimension that has `ArgLength = noarr::dynamic_arg_length`.
It should not contain any `noarr::length_in<Dim>` for dimensions of `noarr::dynamic_arg_length` and `noarr::static_arg_length`.
In some cases, these items can be omitted.
For example, indices are not needed when computing the size, or when computing an index in an unrelated branch of tuple.

During structure manipulations, the signature is in a way dual to the state.
Signature is built in the innermost structures and then retrieved and gradually modified by the outer layers of the structure.
Conversely, the state is passed by the caller to the outermost structure, which updates it and passes it on to the inner layers.

When a structure *adds* a dimension to the signature (as in the first [example](#examples)) during building,
it usually *consumes* the same dimension from the state during offset calculation.
On the other hand, when a structure *masks (removes)* a dimension from the signature during building,
it promises to *produce* the index in the state during offset calculation.
When a structure renames *X to Y* in the signature, it must rename *Y to X* in the state
(because the inner structure is created with *X*, not *Y*, but the interface now mentions *Y* and not *X*).


## Usages

Structure classes make a best-effort attempt to report problems like duplicate dimensions as early as possible.
Signatures are used to check whether the passed [sub-structure](Glossary.md#sub-structure) satisfies the requirements.

[Traverser](Traverser.md) uses the signature to guide the traversal in `for_each`
and to detect [dimension kind](DimensionKinds.md) in `for_dims`.

[Structure traits](other/StructureTraits.md) are implemented using signatures.
A notable example is `scalar_t`, which is used to return the correct type
when accessing a structure element from [`get_at`](BasicUsage.md#get_at) and [bag operator `[]`](BasicUsage.md#indexing-a-bag).

# State

A collection of parameters that can be used for structure queries and in other places (see [Usages](#usages)).
A state item consists of a *tag* and a *value*. Currently, the recognized tags are:
- `noarr::index_in<Dim>` (e.g. `noarr::index_in<'x'>`) gives the [index](Glossary.md#index) for the named [dimension](Glossary.md#dimension)
- `noarr::length_in<Dim>` specifies the [length](Glossary.md#length) of the structure in the named dimension (note: the length must not have been previously specified)
- `noarr::cuda_stripe_index` is specific to [`noarr::cuda_striped`](structs/cuda_striped.md) - see there for more information

The value of a state item can be static (known at compile time and taking no space) or dynamic (only known at runtime) --
for more information, see [Dimension Kinds](DimensionKinds.md).


## Usages

In general, state is used everywhere a set of indices is needed. Usages in noarr are listed below.

### Structure queries

The primary use of state is as a parameter to [functions](BasicUsage.md#functions) and [bag methods](BasicUsage.md#bag):

| use with [structure](Glossary.md#structure) or [bag](Glossary.md#bag) | bag-specific shortcut                      |
| --------------------------------------------------------------------- | ------------------------------------------ |
| `structure_or_bag \| noarr::get_at(data_ptr, state)`                  | `bag[state]`                               |
| `structure_or_bag \| noarr::offset(state)`                            | `bag.offset(state)`                        |
| `structure_or_bag \| noarr::offset<SubStructure>(state)`              | `bag.template offset<SubStructure>(state)` |
| `structure_or_bag \| noarr::get_size(state)`                          | N/A (`bag.get_size()` never needs a state) |
| `structure_or_bag \| noarr::get_length<Dim>(state)`                   | `bag.template get_length<Dim>(state)`      |

State is also used internally when handling these queries.
The top-level structure receives a state from the caller, obtains (and deletes) the items it needs,
and passes the updated state to its [sub-structure(s)](Glossary.md#sub-structure).

The state items expected by a structure are described by the structure's [signature](Signature.md).
The operations carried out on the state during query roughly correspond to those carried out on the signature during structure construction,
as detailed in [the relevant section in Signature documentation](Signature.md#relation-to-state)).

### Traverser

State is the parameter passed by [Traverser](Traverser.md) to the user lambda in [`.for_each(lambda)`](Traverser.md#for_eachlambda).
The other method, [`.template for_dims<...>(lambda)`](Traverser.md#for_dimslambda) does not pass a state,
but [`.state()`](Traverser.md#state-obtaining-a-plain-state-in-for_dims) can be used on the parameter to get one.

### Creating a state

Noarr provides the following functions to create a state instance:

```cpp
using noarr::lit;

// indices in (x, y) are (3, 4)
auto s1 = noarr::make_state<noarr::index_in<'x'>, noarr::index_in<'y'>>(3, 4);

// shortcut for the above
auto s2 = noarr::idx<'x', 'y'>(3, 4);

// this one also sets the length in one of the dimensions
// - use this in case the structure does not have the length already
// - no shortcut is available
auto s3 = noarr::make_state<noarr::index_in<'x'>, noarr::index_in<'y'>, noarr::length_in<'x'>>(3, 4, 10);

// make the x index static
auto s4 = noarr::idx<'x', 'y'>(lit<3>, 4);

// all combined
auto s5 = noarr::make_state<noarr::index_in<'x'>, noarr::index_in<'y'>, noarr::length_in<'x'>, noarr::cuda_stripe_index>(lit<3>, 4, lit<10>, threadIdx.x + (1<<i));
```

Multiple states can also be merged using the `&` operator.

### Retrieving state items

State items can be retrieved using the state's methods or (somewhat more conveniently) using some shortcut functions:

```cpp
auto my_state = s5; // or s1 or s2 or s3 or s4, from the previous snippet
using my_state_t = decltype(my_state);

// the easy way to get indices
auto x1 = noarr::get_index<'x'>(my_state);
auto [x2, y2] = noarr::get_indices<'x', 'y'>(my_state);

// the hard way to get (not only) indices
auto x3 = my_state.template get<noarr::index_in<'x'>>();
auto xlen = my_state.template get<noarr::length_in<'x'>>();

// getting types - might spare you one decltype
using x_t = noarr::state_get_t<my_state_t, noarr::index_in<'x'>>;

// only works with static state item values (e.g. s5)
constexpr std::size_t sx = x_t::value;
constexpr std::size_t sxlen = noarr::state_get_t<my_state_t, noarr::length_in<'x'>>::value;

// query the existence of item
constexpr bool has_xlen = my_state_t::template contains<noarr::length_in<'x'>>;
if constexpr(has_xlen) { /*...*/ } /*...*/
```

### Updating a state

State is immutable, you have to create a new state:

```cpp
// increment index in 'x' and return a new state (other indices are copied unchanged)
auto new_state = noarr::update_index<'x'>(my_state, [](auto x) {return x + 1;});

// shortcut for the above, can update more than one index, but cannot work with lambdas
auto state_east = noarr::neighbor<'x'>(my_state, 1);

auto state_north_east = noarr::neighbor<'x', 'y'>(my_state, +1, -1);
auto state_south_east = noarr::neighbor<'x', 'y'>(my_state, +1, +1);

// create a new item (or replace existing one without seeing it)
auto my_state_3d = my_state.template with<noarr::index_in<'z'>>(0);
```

### Fixing a state in a structure

A state consisting of just indices can be used as argument to [`noarr::fix`](structs/fix.md).
As a result, the dimensions in the structure for which indices are specified in the state are fixed.

For example:

```cpp
auto structure = noarr::scalar<float>() ^ noarr::array<'x', 10>() ^ noarr::array<'y', 20>() ^ noarr::array<'z', 30>();

auto xy = noarr::idx<'x', 'y'>(3, 4);

auto fixed = structure ^ noarr::fix(xy);

auto item_3_4_5 = fixed | noarr::offset(noarr::idx<'z'>(5));
auto item_3_4_6 = fixed | noarr::offset<'z'>(6);
```

Normally the state would be created somewhere else (e.g. given to you in [traverser](Traverser.md) lambda).
Otherwise, you could just use `... ^ noarr::fix<'x', 'y'>(3, 4)`.

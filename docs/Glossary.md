# Glossary

### `^` (operator)

The `^` operator can be used to compose [structures](#structure) and [proto-structures](#proto-structure) into larger (proto-)structures.
It was chosen for its visual similarity with a common symbol for exponentiation.

### `|` (operator)

The `|` operator can be used to apply a [function](BasicUsage.md#functions) to a [structure](#structure). The structure goes to the left-hand side and the function to the right-hand side.
This allows a more fluent "word order" without requiring all the structures to declare and define all the functions.
The `|` operator was chosen for its visual similarity with the shell pipeline syntax.

### bag

An object that remembers both the structure [structure](#structure) and a pointer to data. See [the section on bag in Basic Usage](BasicUsage.md#bag).

### dimension

A [structure](#structure) can have multiple dimensions. Unlike multidimensional arrays in C++ or C, these dimensions are named -- identified by `char`s (e.g. `'x'`, `'y'`).
Like in C++ or C arrays, to get the memory offset of an element, one must specify the [index](#index) for each dimension.
The [lengths](#length) (index ranges) can be [queried/set](BasicUsage.md#lengths) on the structure,
and the dimensions themselves can be inspected at compile time using the structure's [signature](#signature).
See also [Dimension Kinds](DimensionKinds.md).

### dynamic value

A value ([index](#index), [length](#length), [state](#state) item, [structure](#structure) property, ...) that is *not* known at compile time. Represented using `std::size_t`.
See [Dimension Kinds](DimensionKinds.md).

### index

The index in a [dimension](#dimension) specifies the position in a [structure](#structure) with respect to that dimension.
Indices always range from zero (inclusive) to the [length](#length) in the respective dimension (exclusive).
Some dimensions may require the index to be known at compile time - see [Dimension Kinds](DimensionKinds.md) for more information.

### length

The length in a [dimension](#dimension) is a number delimiting the range of valid [indices](#index) in that dimension.
The length in a dimension is a property of a [structure](#structure). A structure with three dimensions has three separate lengths. Do not confuse with [size](#size).
See [the section on lengths in Basic Usage](BasicUsage.md#lengths).

### offset

The offset of an element within a [structure](#structure) is the number of bytes (not elements) from the start of the structure to the start of the element.
It can be queried using [`offset`](BasicUsage.md#offset).

### proto-structure

A proto-structure is similar to a [structure](#structure) template. It cannot be used on its own until it is instantiated, i.e. until another structure is passed to it as an argument.
See [Defining Structures](DefiningStructures.md) for more information about how proto-structures are created and used.

### signature

The signature of a [structure](#structure) describes the interface of the structure: the [dimension](#dimension) names, their order, their [kind](DimensionKinds.md), and the element type.
Signatures are purely compile-time objects (represented by C++ types without values) and can be obtained from the `::signature` member typedef of structure types.
See [Signature](Signature.md) for more information.

### size

The number of bytes the data occupy. Size is a property of a [structure](#structure). Do not confuse with [length](#length).
The size can be queried using [`get_size`](BasicUsage.md#get_size).

### state

A collection of [indices](#index) and potentially other parameters needed to query a property of a [structure](#structure).
See [State documentation](State.md).

### static value

A value ([index](#index), [length](#length), [state](#state) item, [structure](#structure) property, ...) that is known at compile time. Represented using `std::integral_constant<std::size_t, *>`.
See [Dimension Kinds](DimensionKinds.md).

### structure

An object that describes the mapping between [indices](#index) and memory. It can be [queried for](BasicUsage.md#functions) offsets of elements, size in memory, and valid index ranges.
The structure itself does not hold the elements (see [bag](BasicUsage.md#bag) for that). See [Defining Structures](DefiningStructures.md) for more information about how structures are created.

### sub-structure

[Structures](#structure) are often defined in terms of other structures. A sub-structure U of a structure T is any structure used in the definition of T.
Memory that is laid out according to T will usually contain parts that are laid out according to U.
There could possibly be more copies of U, or the layout of T could possibly consist exactly of the layout of V.

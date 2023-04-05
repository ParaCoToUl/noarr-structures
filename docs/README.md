# User documentation for Noarr Structures

Noarr Structures is a header-only library that facilitates the creation of many data structures and provides a layout-agnostic way of accessing the values stored in them.


## Data modeling in Noarr

Data modeling is the process of describing the structure of your data so that an algorithm can be written to processes the data.
*Noarr structures* were designed mainly as a way to model uniform data (as opposed to jagged data).
Uniform data has the advantage of occupying one continuous array of memory. When working with it, you work with three objects:

1. **Structure:** A small, tree-like object, that represents the structure of the data. It does not contain the data itself, nor a pointer to the data.
   It can be thought of as a function that maps indices to memory offsets (in bytes). It stores information, such as data dimensions and tuple types.
   See [Basic Usage](BasicUsage.md) and especially [Defining Structures](DefiningStructures.md) for more information.
2. **Data:** A continuous block of bytes that contains the actual data. Its structure is defined by a corresponding *Structure* object.
3. **Bag:** Wrapper object, which combines *structure* and *data* together. See [Basic Usage](BasicUsage.md) for more information.


## Algorithm implementation

Optionally, you can use the fourth component:

4. **Traverser:** A temporary object that describes the subset of structure's elements and the order in which they should be traversed.
   See [Traverser](Traverser.md) for more information.

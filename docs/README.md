# User documentation for Noarr Structures

## Data modeling

Data modeling is the process of describing the structure of your data so that an algorithm can be written to processes the data. Noarr lets you model your data in an abstract, multidimensional space, abstracting away any underlying physical structure.

## Data modeling in Noarr

*Noarr structures* were designed to support uniform data. Uniform data has the advantage of occupying one continuous array of memory. When working with it, you work with three objects:

1. **Structure:** A small, tree-like object, that represents the structure of the data. It does not contain the data itself, nor a pointer to the data. It can be thought of as a function that maps indices to memory offsets (in bytes). It stores information, such as data dimensions and tuple types.
2. **Data:** A continuous block of bytes that contains the actual data. Its structure is defined by a corresponding *Structure* object.
3. **Bag:** Wrapper object, which combines *structure* and *data* together.

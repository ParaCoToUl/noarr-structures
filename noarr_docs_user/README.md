# User documentation for Noarr

Noarr framework aims to help with certain aspects of performant GPU algorithm development:

1. [Data modelling](#data-modelling)
2. Data serialization
3. Algorithm benchmarking and optimization
4. Algorithm packaging (exporting a library for C++, Python and R)


<a name="data-modelling"></a>
## Data modelling

Data modelling is the process of describing the structure of your data, so that an algorithm can be written to processes the data. Noarr lets you model your data in an abstract, multidimensional space, abstracting away any underlying physical structure.

Noarr framework distinguishes two types of mutidimensional data - smooth and jagged.

**Jagged data** can be thought of as a vector of vectors, each having different size. This means the dimensions of such data need to be stored within the data itself, requiring the use of pointers and making processing of such data inefficient. Noarr supports this type of data only at the highest abstraction levels of your data model.

**Smooth data** can be though of as a multidimensional cube of values. It's like a vector of same-sized vectors, but it also supports tuples and other structures. This lets us store the dimensions separately from the data, letting us freely change the order of specification of dimensions - completely separating the physical data layout from the data model.


<a name="smooth-data-modelling"></a>
### Smooth data modelling

TODO

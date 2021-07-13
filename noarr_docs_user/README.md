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

Smooth data has the advantage of occupying one continuous stretch of memory. When working with it, you work with two object:

1. **Structures:** A small, tree-like object, that represents the structure of the data. It doesn't contain the data itself, nor a pointer to the data. It can be thought of as a function that maps indices to memory offsets (in bytes). It stores information, such as data dimensions and tuple types.
2. **Blobs:** A continuous block of bytes that contains the actual data. Its structure is defined by a corresponding *structure* object.


#### Creating a structure

To represent a list of floats, you create the following *structure* object:

```cpp
noarr::vector<'i', noarr::scalar<float>> my_structure;
```

The only dimension of this structure has the label `i` and it has to be specified in order to access individual scalar values. But currently the structure has no size, we need to make room for 10 items:

```cpp
auto my_structure_of_ten = my_structure % noarr::resize<'i'>{10};
```

A *structure* object is immutable. The `%` operator (the pipe) is used to create modified variants of *structures*. You can chain such operations to arrive at the structure that represents your data.

> The pipe operator is the preffered way to query or modify structures, as it automatically locates the proper sub-structure with the given dimension label.

The reason we specify the size later is that it allows us to decouple the *structure* layout from the resizing action. The resizing action specifies a dimension label `i` and it doesn't care, where that dimension is inside the *structure*.

Here's how we would create a vector of arrays, that can either be in SoA or AoS, based on a constant we can vary to benchmark different physical layouts:

```cpp
const int POLICY_SOA = 1;
const int POLICY_AOS = 2;

const int USED_POLICY = POLICY_SOA; // tweak for benchmarking

template<int Policy>
auto create_structure() {
    if (Policy == POLICY_AOS) {
        return noarr::vector<'v', noarr::array<'a', 5, noarr::scalar<float>>>{};
    }
    if (Policy == POLICY_SOA) {
        return noarr::array<'a', 5, noarr::vector<'v', noarr:scalar<float>>>{};
    }
}

int main() {
    std::size_t data_size = 42; // obtain size from somewhere

    // create a structure of proper dimensions
    auto my_structure = create_structure<USED_POLICY>()
        % noarr::resize<'v'>{size};
        // you could set sizes of all dynamic dimensions here;
    
    // ... work with the structure ...
}
```


#### Allocating and accessing a bag

Now that we have a structure defined, we can create a bag to store the data:

```cpp
// we have our structure
auto my_structure = create_my_structure();

int size = 1024;
auto bag = noarr::bag(noarr::wrap(my_structure).template set_length<'x'>(size).template set_length<'y'>(size));
```

Now, with a blob that holds the values, we can access these values by computing their offset in the blob:

```cpp
// get the reference
std::size_t value_ref = bag.layout().template get_at<'x', 'y'>(bag.data(), i, j);

// now use the reference to access the value
value_ref = 42f;
```

The `noarr::fix` functor transforms the structure into one, which has the corresponding dimension fixed (it's a way to specify an index for a dimension). The `noarr::offset` functor then returns the offset (in bytes) in the blob, where the corresponding scalar value is located (all dimensions have to be fixed by now).

> **TODO:** There will be a better way to access values, that doesn't involve direct pointer arithmetic.

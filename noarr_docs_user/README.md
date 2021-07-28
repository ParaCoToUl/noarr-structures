# User documentation for Noarr

~~Noarr framework aims to help with certain aspects of performant GPU algorithm development:

~~1. [Data modelling](#data-modelling)
~~2. Data serialization
~~3. Algorithm benchmarking and optimization
~4. Algorithm packaging (exporting a library for C++, Python and R)


<a name="data-modelling"></a>
## Data modelling

Data modelling is the process of describing the structure of your data, so that an algorithm can be written to processes the data. Noarr lets you model your data in an abstract, multidimensional space, abstracting away any underlying physical structure.

Noarr framework distinguishes two types of mutidimensional data - uniform and jagged.

**Jagged data** can be thought of as a vector of vectors, each having different size. This means the dimensions of such data need to be stored within the data itself, requiring the use of pointers and making processing of such data inefficient. Noarr supports this type of data only at the highest abstraction levels of your data model.

**Uniform data** can be though of as a multidimensional cube of values. It's like a vector of same-sized vectors, but it also supports tuples and other structures. This lets us store the dimensions separately from the data, letting us freely change the order of specification of dimensions - completely separating the physical data layout from the data model.


<a name="data-modelling-in-noarr"></a>
### Data modelling in Noarr

*Noarr structures* was designed to support uniform data. Uniform data has the advantage of occupying one continuous stretch of memory. When working with it, you work with three objects:

1. **Structure:** A small, tree-like object, that represents the structure of the data. It doesn't contain the data itself, nor a pointer to the data. It can be thought of as a function that maps indices to memory offsets (in bytes). It stores information, such as data dimensions and tuple types.
2. **Data:** A continuous block of bytes that contains the actual data. Its structure is defined by a corresponding *Structure* object.
3. **Bag:** Wraper object, which combines *structure* and *data* together.

> Note: in the case of jagged data, you can use *Noarr pipelines* without *Noarr structures*. The architecture of the GPU is designed for uniform data mainly, so it should fit most common cases. Also note, that you can also use several *Noarr structures* in your program.

#### Creating a structure

To represent a list of floats, you create the following *structure* object:

```cpp
noarr::vector<'i', noarr::scalar<float>> my_structure;
```

The only dimension of this *structure* has the label `i` and it has to be specified in order to access individual scalar values. But currently the structure has no size, we need to make room for 10 items:

```cpp
auto my_structure_of_ten = my_structure | noarr::resize<'i'>(10);
```

A *structure* object is immutable. The `|` operator (the pipe) is used to create modified variants of *structures*. You can chain such operations to arrive at the structure that represents your data.

> The pipe operator is the preferred way to query or modify structures, as it automatically locates the proper sub-structure with the given dimension label (in compile time).

The reason we specify the size later is that it allows us to decouple the *structure* structure from the resizing action. The resizing action specifies a dimension label `i` and it doesn't care, where that dimension is inside the *structure*.

<a name="wrapper"></a>
#### Wrapper
It is possible to use `.` (dot) instead of `|` (pipe), but you have to use `naorr::wrapper` first.

TODO




#### Allocating and accessing *data* and *bag*

Now that we have a structure defined, we can create a bag to store the data. Bag allocates *data* buffer automatically:

```cpp
// we will create a bag
auto bag = noarr::bag(my_structure_of_ten);
```


Now, with a *data* that holds the values, we can access these values by computing their offset in the *bag*:

```cpp
// get the reference (we will get 5-th element)
float& value_ref = bag.structure().template get_at<'i'>(bag.data(), 5);

// now use the reference to access the value
value_ref = 42f;
```

<a name="changing-data-layouts"></a>
#### Changing data layout (*structure*)

Now we want to change the data layout. Noarr needs to know the structure at compile time (for performance). So the right approach is to template all functions and then select between compiled versions. We define different structures like this:

```cpp
// layout declaration
using matrix_rows = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
using matrix_columns = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
```

We will create a templated matrix. And also set size at runtime like this:

```cpp
// function which does some logic templated by different structures
template<typename Structure>
void matrix_demo(int size) {
	// dot version
	auto n1 = noarr::bag(noarr::wrap(Structure()).template set_length<'x'>(size).template set_length<'y'>(size));
	// pipe version (both are valid syntax and produce the same result)
	auto n2 = noarr::bag(Structure() | noarr::set_length<'x'>(size) | noarr::set_length<'y'>(size));
}
```

We set the size at runtime because size can be any int.

We can call at runtime different templated layouts.

```cpp
// we select the layout in runtime
void main() {
	int layout;
	std::cin >> layout;
	
	if (layout == 1)
		matrix_demo<matrix_rows>(size);
	else if (layout == 2)
		matrix_demo<matrix_columns>(size);
	// and so on
}
```

<a name="supported-layouts"></a>
#### Our supported layouts (*structures*)
##### Containers

Noarr is designed to be easily extendable, we implemented basic ones and some simple 2D layouts.

```cpp
noarr::vector<'i', noarr::scalar<float>> my_vector;
noarr::array<'i', 10, noarr::scalar<float>> my_array;
```

##### Scalars

Noarr supports all scalars, for example: `bool`, `int`, `char`, `float`, `double`, `long`, `std::size_t`...

##### Tuples

We declare tuple like this:

```cpp
noarr::tuple<'t', noarr::scalar<int>, noarr::scalar<float>> t;
noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'x', noarr::scalar<int>>> t2;
noarr::tuple<'t', noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>>, noarr::vector<'x', noarr::array<'y', 20, noarr::scalar<int>>>> t3;
```

To get the first element of the tuple we use `get_at` in the following way:

```cpp
get_at<'t'>(1_idx);
```


##### Matrices/Cubes

We will shortly discuss higher-dimensional data. You will model the matrix in the following way:

```cpp
noarr::vector<'i', noarr::vector<'j', noarr::scalar<float>>> my_matrix;
```

To showcase easy extendability we implemented Z-curve and block layout:

```cpp
noarr::zcurve<'i', 'j', noarr::scalar<float>>> my_zcurve_matrix;
```

We can use `get_at<>` in the following ways

```cpp
get_at<'i', 'j'>(1, 2);
```

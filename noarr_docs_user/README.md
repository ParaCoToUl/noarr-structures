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

1. **Structure:** A small, tree-like object, that represents the structure of the data. It doesn't contain the data itself, nor a pointer to the data. It can be thought of as a function that maps indices to memory offsets (in bytes). It stores information, such as data dimensions and tuple types.
2. **Data:** A continuous block of bytes that contains the actual data. Its structure is defined by a corresponding *Structure* object.
3. **Bag:** Wraper object, which combines *structure* and *data* together.

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

> The pipe operator is the preffered way to query or modify structures, as it automatically locates the proper sub-structure with the given dimension label (in compile time).

The reason we specify the size later is that it allows us to decouple the *structure* structure from the resizing action. The resizing action specifies a dimension label `i` and it doesn't care, where that dimension is inside the *structure*.



#### Allocating and accessing *data* and *bag*

Now that we have a structure defined, we can create a bag to store the data:

```cpp
// we will create a bag
auto bag = noarr::bag(noarr::wrap(my_structure_of_ten));
```


Now, with a *data* that holds the values, we can access these values by computing their offset in the *bag*:

```cpp
// get the reference (we will get 5-th element)
std::size_t value_ref = bag.structure().template get_at<'i'>(bag.data(), 5);

// now use the reference to access the value
value_ref = 42f;
```


#### Changing data layout (*structure*)

Now we want to change the data layout. Noarr neeeds to know the structure in compile time (for performance). So the right approach is to template all funtions and then select between compiled versions. We define diferent structures like this:

```cpp
enum MatrixDataLayout { Rows = 0, Columns = 1, Zcurve = 2 };

template<MatrixDataLayout layout>
struct GetMatrixStructreStructure;

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Rows> {
	static constexpr auto GetMatrixStructure() {
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Columns> {
	static constexpr auto GetMatrixStructure() {
		return noarr::vector<'y', noarr::vector<'x', noarr::scalar<int>>>();
	}
};

template<>
struct GetMatrixStructreStructure<MatrixDataLayout::Zcurve> {
	static constexpr auto GetMatrixStructure() {
		return noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>();
	}
};
```

We will create templated matrix. And also set size in runtime like this:

```cpp
template<MatrixDataLayout layout>
void matrix_template_test(int size) {
	auto m1 = noarr::bag(noarr::wrap(GetMatrixStructreStructure<layout>::GetMatrixStructure()).template set_length<'x'>(size).template set_length<'y'>(size));
}
```

We set the size in runtime, because size can be any int.

We can calling runtime different templated layouts.

```cpp
void matrix_template_test_runtime(MatrixDataLayout layout, int size)
{
	if (layout == MatrixDataLayout::Rows)
		matrix_template_test<MatrixDataLayout::Rows>(size);
	else if (layout == MatrixDataLayout::Columns)
		matrix_template_test<MatrixDataLayout::Columns>(size);
	else if (layout == MatrixDataLayout::Zcurve)
		matrix_template_test<MatrixDataLayout::Zcurve>(size);
}
```
#### Our supported layouts (*structures*)

Noarr is designed to be easily extandable, we implemented basic ones and some simple 2D layouts.

```cpp
noarr::vector<'i', noarr::scalar<float>> my_vector;
noarr::vector<'i', 10, noarr::scalar<float>> my_array; //TODO: ???
```

You will model matrix in a following way:

```cpp
noarr::vector<'i', noarr::vector<'j', noarr::scalar<float>>> my_matrix;
```

To showcase easy extendability we implemented Z-curve and block layout:

```cpp
noarr::zcurve<'i', 'j', noarr::scalar<float>>> my_zcurve_matrix; //TODO: ???
```


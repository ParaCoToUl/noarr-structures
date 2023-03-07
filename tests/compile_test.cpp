/**
 * @file compile_test.cpp
 * @brief This file contains the code snippets from the root README.md ensuring they are correct
 * 
 */

#include <catch2/catch_test_macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>


TEST_CASE("Main example compile test", "[Main example compile test]") {
	// the following two structures both describe a two-dimensional continuous array (matrix)

	// describes a layout of 20x30 two-dimensional array
	noarr::array<'x', 20, noarr::array<'y', 30, noarr::scalar<int>>> foo;

	// describes a similar logical layout with switched dimensions in the physical layout
	noarr::array<'y', 30, noarr::array<'x', 20, noarr::scalar<int>>> bar;

	// getting the offset of the value at (x = 5; y = 10):
	foo | noarr::offset<'x', 'y'>(5, 10);
	bar | noarr::offset<'x', 'y'>(5, 10);

	// arguments definition
	int WIDTH = 10;
	int HEIGHT = 10;

	// defines the structure of the matrix, rows are the 'x' dimension and columns are the 'y' dimension
	// physically, the layout is an contiguous array of rows
	noarr::vector<'y', noarr::vector<'x', noarr::scalar<int>>> matrix_structure;

	// defining size of the matrix
	auto sized_matrix_structure = matrix_structure ^ noarr::set_length<'x'>(WIDTH) ^ noarr::set_length<'y'>(HEIGHT);

	// data allocation
	auto matrix = noarr::make_bag(sized_matrix_structure);

	for (std::size_t i = 0; i < matrix.get_length<'x'>(); i++)
		for (std::size_t j = i; j < matrix.get_length<'y'>(); j++)
			std::swap(matrix.at<'x', 'y'>(i, j), matrix.at<'x', 'y'>(j, i));
}


// function which does some logic templated by different structures
template<typename Structure>
void matrix_demo(int size) {
	noarr::make_bag(Structure() ^ noarr::set_length<'x'>(size) ^ noarr::set_length<'y'>(size));
}

TEST_CASE("Example compile test", "[Example compile test]") {
	noarr::vector<'i', noarr::scalar<float>> my_structure;
	auto my_structure_of_ten = my_structure ^ noarr::set_length<'i'>(10);

	// we will create a bag
	auto bag = noarr::make_bag(my_structure_of_ten);

	// get the reference (we will get 5-th element)
	float& value_ref = bag.structure() | noarr::get_at<'i'>(bag.data(), 5);

	// now use the reference to access the value
	value_ref = 42;

	bag.at<'i'>(5) = 42;

	// layout declaration
	using matrix_rows = noarr::vector<'y', noarr::vector<'x', noarr::scalar<int>>>;
	using matrix_columns = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;

	// arguments definition
	std::string layout = "rows";
	int size = 42;

	// we select the layout in runtime
	if (layout == "rows")
		matrix_demo<matrix_rows>(size);
	else if (layout == "columns")
		matrix_demo<matrix_columns>(size);

	noarr::vector<'i', noarr::scalar<float>> my_vector;
	noarr::array<'i', 10, noarr::scalar<float>> my_array;

	noarr::tuple<'t', noarr::scalar<int>, noarr::scalar<float>> t;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'x', noarr::scalar<int>>> t2;
	noarr::tuple<'t', noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>>, noarr::vector<'x', noarr::array<'y', 20, noarr::scalar<int>>>> t3;

	// to remove warnings:
	my_vector = {};
	my_array = {};
	t = {};
	t2 = {};
	t3 = {};

	// tuple declaration
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::array<'x', 20, noarr::scalar<int>>> tuple;
	// we will create a bag
	auto tuple_bag = noarr::make_bag(tuple);
	// we index tuple like this
	float& value = tuple_bag.at<'t', 'x'>(noarr::lit<0>, 1);
	value = 0.f;
}

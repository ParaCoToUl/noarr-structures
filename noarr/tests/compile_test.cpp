#include <catch2/catch.hpp>

#include "noarr/structures.hpp"
#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/wrapper.hpp"
#include "noarr/structures/bag.hpp"

// function which does some logic templated by different structures
template<typename Structure>
void matrix_demo(int size) {
	// dot version
	auto n1 = noarr::bag(noarr::wrap(Structure()).template set_length<'x'>(size).template set_length<'y'>(size));
	// pipe version (both are valid syntax and produce the same result)
	auto n2 = noarr::bag(Structure() | noarr::set_length<'x'>(size) | noarr::set_length<'y'>(size));
}

TEST_CASE("Example compile test", "[Example compile test]") {
	noarr::vector<'i', noarr::scalar<float>> my_structure;
	auto my_structure_of_ten = my_structure | noarr::set_length<'i'>(10);
	// artificially complicated example
	auto piped = my_structure_of_ten | noarr::set_length<'i'>(5) | noarr::set_length<'i'>(10);
	// now version with wrapper
	auto doted = noarr::wrap(my_structure_of_ten).set_length<'i'>(5).set_length<'i'>(10);

	// we will create a bag
	auto bag = noarr::bag(my_structure_of_ten);

	// get the reference (we will get 5-th element)
	float& value_ref = bag.structure().template get_at<'i'>(bag.data(), 5);

	// now use the reference to access the value
	value_ref = 42;

	bag.at<'i'>(5) = 42;

	// layout declaration
	using matrix_rows = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;
	using matrix_columns = noarr::vector<'x', noarr::vector<'y', noarr::scalar<int>>>;

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

	//t3.get_at<'t'>(1_idx);

	noarr::vector<'i', noarr::vector<'j', noarr::scalar<float>>> my_matrix;

}

#include <catch2/catch.hpp>
//#include "noarr/structures.hpp"

#include <iostream>
#include <array>

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"

using namespace noarr;

TEST_CASE("Image", "[image]") {

	array<'x', 1920, array<'y', 1080, tuple<'p', scalar<float>, scalar<float>, scalar<float>, scalar<float>>>> image;
	array<'x', 1920, array<'y', 1080, array<'z', 4, scalar<float>>>> grayscale;

	SECTION("check is_cube") {
		REQUIRE(is_cube<decltype(grayscale)>::value);
		REQUIRE(!is_cube<decltype(image)>::value);
	}

	SECTION("check TODO") {
		auto value_ref = image | fix<'x'>(0) | fix<'y'>(0) | fix<'p'>(2);
		//float& value_ref = image | fix<'x'>(0) | fix<'y'>(0) | fix<'p'>(2) | offset();
	}
}


/*TEST_CASE("Image", "[image]") {
	std::array<float, 300> data;

	vector<'x', scalar<float>> v;
	array<'y', 20000, vector<'x', scalar<float>>> v2;
	tuple<'t', array<'x', 10, scalar<float>>, vector<'x', scalar<int>>> t;
	tuple<'t', array<'y', 20000, vector<'x', scalar<float>>>, vector<'x', array<'y', 20, scalar<int>>>> t2;

	SECTION("check is_cube") {
		REQUIRE(!is_cube<decltype(v)>::value);
		REQUIRE(!is_cube<decltype(t)>::value);
		REQUIRE(!is_cube<decltype(t2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(v)>::value);
		REQUIRE(std::is_pod<decltype(v2)>::value);
		REQUIRE(std::is_pod<decltype(t)>::value);
	}
}*/

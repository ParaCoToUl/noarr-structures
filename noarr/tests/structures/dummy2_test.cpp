#include <catch2/catch.hpp>
//#include "noarr/structures.hpp"

#include <iostream>
#include <array>

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"

TEST_CASE("Image", "[image]") {

	//noarr::array<'x', 1920, noarr::array<'y', 1080, noarr::tuple<'p', noarr::scalar<float>, noarr::scalar<float>, noarr::scalar<float>, noarr::scalar<float>>>> image;

	noarr::array<'x', 1920, noarr::array<'y', 1080, noarr::array<'p', 4, noarr::scalar<float>>>> grayscale;
	std::vector<char> my_blob_buffer(grayscale | noarr::get_size());
	char* my_blob = (char*)my_blob_buffer.data();

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(grayscale)>::value);
		//REQUIRE(!noarr::is_cube<decltype(image)>::value);
	}

	SECTION("check TODO") {
		//auto value_ref = image | fix<'x'>(0) | fix<'y'>(0) | fix<'p'>(2);
		//std::declval

		//REQUIRE((typeid(image | noarr::fix<'x'>(0) | noarr::fix<'y'>(0) | noarr::fix<'p'>(2)).name()) == "float");
		//auto value_ref = image | noarr::fix<'x'>(0) | noarr::fix<'y'>(0) | noarr::fix<'p'>(2);
		//std::size_t offset = image | noarr::fixs<'x', 'y', 'p'>(0, 0, 2) | noarr::offset();
		std::size_t offset = grayscale | noarr::fix<'x'>(0) | noarr::fix<'y'>(0) | noarr::fix<'p'>(2) | noarr::offset();
		float& value_ref = *((float*)(my_blob + offset));
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



TEST_CASE("Vector", "[resizing]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto v2 = v | noarr::resize<'x'>(10); // transform

	SECTION("size check 1") {
		REQUIRE((v2 | noarr::get_size()) == 10);
	}

	auto v3 = v | noarr::resize<'x'>(20); // transform
	auto v4 = v2 | noarr::resize<'x'>(30); // transform

	SECTION("size check 2") {
		REQUIRE((v2 | noarr::get_size()) == 10);
		REQUIRE((v3 | noarr::get_size()) == 20);
		REQUIRE((v4 | noarr::get_size()) == 30);
	}

	SECTION("check is_cube 2") {
		REQUIRE(!noarr::is_cube<decltype(v)>::value);
		REQUIRE(noarr::is_cube<decltype(v2)>::value);
		REQUIRE(noarr::is_cube<decltype(v3)>::value);
		REQUIRE(noarr::is_cube<decltype(v4)>::value);
	}

	auto v5 = v4 | noarr::resize<'x'>(-10); // transform

	SECTION("size check 3") {
		REQUIRE((v5 | noarr::get_size()) == -10);
	}

	SECTION("check is_cube 3") {
		REQUIRE(noarr::is_cube<decltype(v5)>::value);
	}
}



TEST_CASE("Vector2", "[is_trivial]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto v2 = v | noarr::resize<'x'>(10); // transform

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = v | noarr::resize<'x'>(20); // transform
	auto v4 = v2 | noarr::resize<'x'>(30); // transform

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}


TEST_CASE("Array", "[is_trivial]")
{
	noarr::array<'x', 1920, noarr::scalar<float>> v;
	auto v2 = v | noarr::resize<'x'>(10); // transform

	SECTION("is_trivial check 1") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);
	}

	auto v3 = v | noarr::resize<'x'>(20); // transform
	auto v4 = v2 | noarr::resize<'x'>(30); // transform

	SECTION("is_trivial check 2") {
		REQUIRE(std::is_trivial<decltype(v2)>::value);
		REQUIRE(std::is_standard_layout<decltype(v2)>::value);

		REQUIRE(std::is_trivial<decltype(v3)>::value);
		REQUIRE(std::is_standard_layout<decltype(v3)>::value);

		REQUIRE(std::is_trivial<decltype(v4)>::value);
		REQUIRE(std::is_standard_layout<decltype(v4)>::value);
	}
}



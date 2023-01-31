#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>
#include <tuple>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/wrapper.hpp>
#include "noarr_test_defs.hpp"

TEST_CASE("Vector resizing", "[resizing]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto w = noarr::wrap(v);
	auto v2 = w.set_length<'x'>(10);
	
	SECTION("size check 1") {
		REQUIRE((v2.get_length<'x'>()) == 10);
	}

	auto v3 = w.set_length<'x'>(20);

	SECTION("size check 2") {
		REQUIRE((v2.get_length<'x'>()) == 10);
		REQUIRE((v3.get_length<'x'>()) == 20);
	}

	SECTION("check is_cube 2") {
		REQUIRE(!noarr::is_cube<decltype(v)>::value);
		REQUIRE(noarr::is_cube<decltype(v2)>::value);
		REQUIRE(noarr::is_cube<decltype(v3)>::value);
	}

	SECTION("size check 3") {
		REQUIRE((v2.get_length<'x'>()) == 10);
		REQUIRE((v3.get_length<'x'>()) == 20);
	}
}

TEST_CASE("Vector2 resizing", "[is_simple]")
{
	noarr::vector<'x', noarr::scalar<float>> v;
	auto w = noarr::wrap(v);
	auto v2 = w.set_length<'x'>(10);

	SECTION("is_simple check 1") {
		REQUIRE(noarr_test::type_is_simple(v2));
	}

	auto v3 = w.set_length<'x'>(20);

	SECTION("is_simple check 2") {
		REQUIRE(noarr_test::type_is_simple(v2));

		REQUIRE(noarr_test::type_is_simple(v3));
	}
}

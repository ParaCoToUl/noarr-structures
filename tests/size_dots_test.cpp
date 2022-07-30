#include <catch2/catch.hpp>

#include <iostream>
#include <array>

#include "noarr/structures_extended.hpp"
#include "noarr_test_defs.hpp"

using namespace noarr::literals;

TEST_CASE("Sizes is_cube is_simple", "[sizes is_cube is_simple]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'y', noarr::scalar<int>>> t;
	noarr::tuple<'t', noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>>, noarr::vector<'a', noarr::array<'b', 20, noarr::scalar<int>>>> t2;

	SECTION("check is_cube") {
		REQUIRE(!noarr::is_cube<decltype(v)>::value);
		REQUIRE(!noarr::is_cube<decltype(t)>::value);
		REQUIRE(!noarr::is_cube<decltype(t2)>::value);
	}

	SECTION("check is_simple") {
		REQUIRE(noarr_test::type_is_simple(v));
		REQUIRE(noarr_test::type_is_simple(v2));
		REQUIRE(noarr_test::type_is_simple(t));
	}
}

TEST_CASE("Sizes sizes", "[sizes sizes]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'y', noarr::scalar<int>>> t;
	noarr::tuple<'t', noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>>, noarr::vector<'a', noarr::array<'b', 20, noarr::scalar<int>>>> t2;
	
	auto v_sized = v | noarr::set_length<'x'>(20);

	auto w_v_sized = noarr::wrap(v_sized);
	auto w_v2 = noarr::wrap(v2);
	auto w_t = noarr::wrap(t);
	auto w_t2 = noarr::wrap(t2);

	SECTION("check cizes") {
		REQUIRE(w_v_sized.get_length<'x'>() == 20);
		REQUIRE(w_v2.get_length<'y'>() == 20000);
		REQUIRE(w_t.get_length<'x'>() == 10);
		REQUIRE(w_t2 .get_length<'y'>() == 20000);
		REQUIRE(w_t2.get_length<'b'>() == 20);
	}
}

TEST_CASE("Resize", "[transform]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	auto w = noarr::wrap(v);

	auto vs = w.set_length<'x'>(10);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs)>::value);
	}

	SECTION("check is_simple") {
		REQUIRE(noarr_test::type_is_simple(vs));
	}

	SECTION("check size") {
		REQUIRE(vs.get_length<'x'>() == 10);
	}
}

TEST_CASE("Resize 2", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto w = noarr::wrap(v2);
	auto vs2 = w.set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_simple") {
		REQUIRE(noarr_test::type_is_simple(vs2));
	}

	SECTION("check is_simple") {
		REQUIRE(noarr_test::type_is_simple(vs2.fix<'y', 'x'>(5, 5)));
		REQUIRE(noarr_test::type_is_simple(noarr::fix<'y', 'x'>(5, 5)));
	}

	SECTION("check size") {
		REQUIRE(vs2.get_length<'x'>() == 20);
		REQUIRE(vs2.get_length<'y'>() == 20000);
	}
}

TEST_CASE("Resize 3", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto w = noarr::wrap(v2);
	auto vs2 = w.set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_simple") {
		REQUIRE(noarr_test::type_is_simple(vs2));
	}

	SECTION("check size") {
		REQUIRE(vs2.get_length<'x'>() == 20);
		REQUIRE(vs2.get_length<'y'>() == 20000);
	}
}

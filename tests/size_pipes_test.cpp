#include <catch2/catch.hpp>

#include <iostream>
#include <array>

#include "noarr/structures_extended.hpp"

using namespace noarr::literals;

TEST_CASE("Pipes Sizes is_cube is_pod", "[sizes is_cube is_pod]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'y', noarr::scalar<int>>> t;
	noarr::tuple<'t', noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>>, noarr::vector<'x', noarr::array<'y', 20, noarr::scalar<int>>>> t2;

	SECTION("check is_cube") {
		REQUIRE(!noarr::is_cube<decltype(v)>::value);
		REQUIRE(!noarr::is_cube<decltype(t)>::value);
		REQUIRE(!noarr::is_cube<decltype(t2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(v)>::value);
		REQUIRE(std::is_pod<decltype(v2)>::value);
		REQUIRE(std::is_pod<decltype(t)>::value);
	}
}

TEST_CASE("Pipes sizes", "[sizes sizes]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'y', noarr::scalar<int>>> t;
	noarr::tuple<'t', noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>>, noarr::vector<'a', noarr::array<'b', 20, noarr::scalar<int>>>> t2;

	SECTION("check cizes") {
		REQUIRE((v2 | noarr::get_length<'y'>()) == 20000);
		REQUIRE((t | noarr::get_length<'x'>()) == 10);
		REQUIRE((t2 | noarr::get_length<'y'>()) == 20000);
		REQUIRE((t2 | noarr::get_length<'b'>()) == 20);
	}
}

TEST_CASE("Pipes Resize", "[transform]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	auto vs = v | noarr::set_length<'x'>(10);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs)>::value);
	}

	SECTION("check size") {
		REQUIRE((vs | noarr::get_length<'x'>()) == 10);
	}
}

TEST_CASE("Pipes Resize 2", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto vs2 = v2 | noarr::set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2 | noarr::fix<'y', 'x'>(5, 5))>::value);
		REQUIRE(std::is_pod<decltype(noarr::fix<'y', 'x'>(5, 5))>::value);
	}

	SECTION("check point") {
		REQUIRE(noarr::is_point<decltype(vs2 | noarr::fix<'y', 'x'>(5, 5))>::value);
	}

	SECTION("check size") {
		REQUIRE((vs2 | noarr::get_length<'x'>()) == 20);
		REQUIRE((vs2 | noarr::get_length<'y'>()) == 20000);
	}
}

TEST_CASE("Pipes Resize 3", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto vs3 = v2 | noarr::set_length<'x'>(10_idx);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs3)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs3)>::value);
	}

	SECTION("check size") {
		REQUIRE((vs3 | noarr::get_length<'x'>()) == 10);
		REQUIRE((vs3 | noarr::get_length<'y'>()) == 20000);
	}
}

TEST_CASE("Pipes Resize 4", "[Resizing]") {
	volatile std::size_t l = 20;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'x', noarr::scalar<int>>> t;
	auto vs4 = pipe(v2, noarr::set_length<'y'>(10_idx), noarr::set_length<'x'>(l));

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs4)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs4)>::value);
	}

	auto ts = t | noarr::set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(!noarr::is_cube<decltype(ts)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(ts)>::value);
	}
}

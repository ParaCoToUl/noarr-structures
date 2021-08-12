#include <catch2/catch.hpp>

#include <iostream>
#include <array>

#include "noarr/structures_extended.hpp"

using namespace noarr::literals;

// TODO: change into a test of whether structures allow various pipes and what should be constexpr is constexpr

TEST_CASE("Sizes", "[sizes]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'x', noarr::scalar<int>>> t;
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

TEST_CASE("Resize", "[transform]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	auto w = noarr::wrap(v);

	auto vs = w.set_length<'x'>(10); // transform

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs)>::value);
	}
}

TEST_CASE("Resize 2", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto w = noarr::wrap(v2);
	auto vs2 = w.set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2.fix<'y', 'x'>(5, 5))>::value);
		REQUIRE(std::is_pod<decltype(noarr::fix<'y', 'x'>(5, 5))>::value);
	}
}

TEST_CASE("Resize 3", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto w = noarr::wrap(v2);
	auto vs2 = w.set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2)>::value);
	}
}

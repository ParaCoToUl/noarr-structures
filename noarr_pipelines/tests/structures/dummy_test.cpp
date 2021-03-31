#include <catch2/catch.hpp>
//#include "noarr/structures.hpp"

#include <iostream>

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"

using namespace noarr;

TEST_CASE("Sizes", "[sizes]") {
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
}

TEST_CASE("Resize", "[transform]") {
	vector<'x', scalar<float>> v;
	auto vs = v | resize<'x'>(10); // transform

	SECTION("check is_cube") {
		REQUIRE(is_cube<decltype(vs)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs)>::value);
	}
}

TEST_CASE("Resize 2", "[Resizing]") {
	array<'y', 20000, vector<'x', scalar<float>>> v2;
	auto vs2 = v2 | resize<'x'>(20); // transform

	SECTION("check is_cube") {
		REQUIRE(is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2)>::value);
	}

	/*typeid(vs2).name();
	vs2.size();
	sizeof(vs2);
	typeid(vs2 | fix<'x'>(5)).name();
	typeid(vs2 | fix<'x'>(5) | fix<'y'>(5)).name();
	typeid(vs2 | fixs<'x', 'y'>(5, 5)).name();
	(vs2 | fixs<'x', 'y'>(5, 5) | offset());
	(vs2 | fixs<'y', 'x'>(5, 5) | offset());

	typeid(vs2 | fixs<'y', 'x'>(5, 5) | get_at((char *)nullptr)).name();*/

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2 | fixs<'y', 'x'>(5, 5))>::value);
		REQUIRE(std::is_pod<decltype(fixs<'y', 'x'>(5, 5))>::value);
	}

	//(vs2 | get_offset<'y'>(5));

	SECTION("check point") {
		REQUIRE(is_point<decltype(vs2 | fixs<'y', 'x'>(5, 5))>::value);
	}
}

TEST_CASE("Resize 3", "[Resizing]") {
	array<'y', 20000, vector<'x', scalar<float>>> v2;
	auto vs2 = v2 | resize<'x'>(20); // transform
	auto vs3 = v2 | cresize<'x', 10>(); // transform

	SECTION("check is_cube") {
		REQUIRE(is_cube<decltype(vs3)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs3)>::value);
	}

	typeid(vs3).name();
	vs3.size();
	sizeof(vs3);
}




TEST_CASE("Resize 3", "[Resizing]") {
	volatile std::size_t l = 20;
	array<'y', 20000, vector<'x', scalar<float>>> v2;
	tuple<'t', array<'x', 10, scalar<float>>, vector<'x', scalar<int>>> t;
	// tuple<'t', array<'y', 20000, vector<'x', scalar<float>>>, vector<'x', array<'y', 20, scalar<int>>>> t2;
	auto vs4 = pipe(v2, cresize<'y', 10>(), resize<'x'>(l)); // transform

	SECTION("check is_cube") {
		REQUIRE(is_cube<decltype(vs4)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs4)>::value);
	}

	typeid(vs4).name();
	vs4.size();
	sizeof(vs4);

	sizeof(t);

	auto ts = t | resize<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(!is_cube<decltype(ts)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(ts)>::value);
	}

	//print_struct(std::cout, ts) << " ts;" << std::endl;
	sizeof(ts);
	ts.size();

	//print_struct(std::cout, t2 | reassemble<'x', 'y'>()) << " t2';" << std::endl;
	//print_struct(std::cout, t2 | resize<'x'>(10) | reassemble<'y', 'x'>()) << " t2'';" << std::endl;
	//print_struct(std::cout, t2 | reassemble<'x', 'x'>()) << " t2;" << std::endl;
}

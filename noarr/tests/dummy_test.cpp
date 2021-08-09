#include <catch2/catch.hpp>

#include <iostream>
#include <array>

#include "noarr/structures_extended.hpp"

using namespace noarr::literals;

// TODO: change into a test of whether structures allow various pipes and what should be constexpr is constexpr

TEST_CASE("Pipes Sizes", "[sizes]") {
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

TEST_CASE("Pipes Resize", "[transform]") {
	noarr::vector<'x', noarr::scalar<float>> v;
	auto vs = v | noarr::set_length<'x'>(10); // transform

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs)>::value);
	}
}

TEST_CASE("Pipes Resize 2", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto vs2 = v2 | noarr::set_length<'x'>(20); // transform

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2)>::value);
	}

	/*typeid(vs2).name();
	vs2.size();
	sizeof(vs2);
	typeid(vs2 | fix<'x'>(5)).name();
	typeid(vs2 | fix<'x'>(5) | fix<'y'>(5)).name();
	typeid(vs2 | fix<'x', 'y'>(5, 5)).name();
	(vs2 | fix<'x', 'y'>(5, 5) | offset());
	(vs2 | fix<'y', 'x'>(5, 5) | offset());

	typeid(vs2 | fix<'y', 'x'>(5, 5) | get_at((char *)nullptr)).name();*/

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2 | noarr::fix<'y', 'x'>(5, 5))>::value);
		REQUIRE(std::is_pod<decltype(noarr::fix<'y', 'x'>(5, 5))>::value);
	}

	//(vs2 | get_offset<'y'>(5));

	SECTION("check point") {
		REQUIRE(noarr::is_point<decltype(vs2 | noarr::fix<'y', 'x'>(5, 5))>::value);
	}
}

TEST_CASE("Pipes Resize 3", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	// auto vs2 = v2 | noarr::set_length<'x'>(20); // transform <- // FIXME: NEVER USED
	auto vs3 = v2 | noarr::set_length<'x'>(10_idx); // transform

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs3)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs3)>::value);
	}

	typeid(vs3).name();
	vs3.size();
	// sizeof(vs3); <- // FIXME: NO EFFECT
}




TEST_CASE("Pipes Resize 4", "[Resizing]") {
	volatile std::size_t l = 20;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'x', noarr::scalar<int>>> t;
	// tuple<'t', array<'y', 20000, vector<'x', scalar<float>>>, vector<'x', array<'y', 20, scalar<int>>>> t2;
	auto vs4 = pipe(v2, noarr::set_length<'y'>(10_idx), noarr::set_length<'x'>(l)); // transform

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs4)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs4)>::value);
	}

	typeid(vs4).name();
	vs4.size();
	// sizeof(vs4); <- // FIXME: NO EFFECT

	// sizeof(t); <- // FIXME: NO EFFECT

	auto ts = t | noarr::set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(!noarr::is_cube<decltype(ts)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(ts)>::value);
	}

	//print_struct(std::cout, ts) << " ts;" << std::endl;
	// sizeof(ts); <- // FIXME: NO EFFECT
	ts.size();

	//print_struct(std::cout, t2 | reassemble<'x', 'y'>()) << " t2';" << std::endl;
	//print_struct(std::cout, t2 | set_length<'x'>(10) | reassemble<'y', 'x'>()) << " t2'';" << std::endl;
	//print_struct(std::cout, t2 | reassemble<'x', 'x'>()) << " t2;" << std::endl;
}


//////////
// Dots //
//////////

TEST_CASE("Sizes", "[sizes]") {
	// std::array<float, 300> data; <- // FIXME: NEVER USED

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

	/*
	typeid(vs2).name();
	vs2.size();
	sizeof(vs2);
	typeid(vs2 | fix<'x'>(5)).name();
	typeid(vs2 | fix<'x'>(5) | fix<'y'>(5)).name();
	typeid(vs2 | fix<'x', 'y'>(5, 5)).name();
	(vs2 | fix<'x', 'y'>(5, 5) | offset());
	(vs2 | fix<'y', 'x'>(5, 5) | offset());

	typeid(vs2 | fix<'y', 'x'>(5, 5) | get_at((char *)nullptr)).name();
	*/

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2.fix<'y', 'x'>(5, 5))>::value);
		REQUIRE(std::is_pod<decltype(noarr::fix<'y', 'x'>(5, 5))>::value);
	}

	//(vs2 | get_offset<'y'>(5));

	SECTION("check point") {
		//REQUIRE(noarr::is_point<decltype(vs2.fix<'y', 'x'>(5, 5))>::value); // TODO
	}
}

TEST_CASE("Resize 3", "[Resizing]") {
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	auto w = noarr::wrap(v2);
	auto vs2 = w.set_length<'x'>(20); // transform
	//auto vs3 = w.cresize<'x', 10>(); // transform

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs2)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs2)>::value);
	}

	//typeid(vs2).name();
	//vs2.size();
	//sizeof(vs2);
}



// TODO
/*
TEST_CASE("Resize 4", "[Resizing]") {
	volatile std::size_t l = 20;
	noarr::array<'y', 20000, noarr::vector<'x', noarr::scalar<float>>> v2;
	noarr::tuple<'t', noarr::array<'x', 10, noarr::scalar<float>>, noarr::vector<'x', noarr::scalar<int>>> t;

	auto w = noarr::wrap(v2);
	auto wt = noarr::wrap(t);

	// tuple<'t', array<'y', 20000, vector<'x', scalar<float>>>, vector<'x', array<'y', 20, scalar<int>>>> t2;
	auto vs4 = pipe(w, noarr::cresize<'y', 10>(), noarr::set_length<'x'>(l)); // transform

	SECTION("check is_cube") {
		REQUIRE(noarr::is_cube<decltype(vs4)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(vs4)>::value);
	}

	typeid(vs4).name();
	vs4.size();
	sizeof(vs4);

	sizeof(t);

	auto ts = wt.set_length<'x'>(20);

	SECTION("check is_cube") {
		REQUIRE(!noarr::is_cube<decltype(ts)>::value);
	}

	SECTION("check is_pod") {
		REQUIRE(std::is_pod<decltype(ts)>::value);
	}

	//print_struct(std::cout, ts) << " ts;" << std::endl;
	sizeof(ts);
	//ts.size();

	//print_struct(std::cout, t2 | reassemble<'x', 'y'>()) << " t2';" << std::endl;
	//print_struct(std::cout, t2 | set_length<'x'>(10) | reassemble<'y', 'x'>()) << " t2'';" << std::endl;
	//print_struct(std::cout, t2 | reassemble<'x', 'x'>()) << " t2;" << std::endl;
}
*/

#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <cstddef>

#include <noarr/structures_extended.hpp>

using noarr::lit;

TEST_CASE("Pipes sizes is_cube is_simple", "[sizes is_cube is_simple]") {
	auto v = noarr::vector_t<'x', noarr::scalar<float>>();
	auto v2 = noarr::array_t<'y', 20000, noarr::vector_t<'x', noarr::scalar<float>>>();
	auto t = noarr::tuple_t<'t', noarr::array_t<'x', 10, noarr::scalar<float>>, noarr::vector_t<'y', noarr::scalar<int>>>();
	auto t2 = noarr::tuple_t<'t', noarr::array_t<'y', 20000, noarr::vector_t<'x', noarr::scalar<float>>>, noarr::vector_t<'x', noarr::array_t<'y', 20, noarr::scalar<int>>>>();

	REQUIRE(!noarr::is_cube<decltype(v)>::value);
	REQUIRE(!noarr::is_cube<decltype(t)>::value);
	REQUIRE(!noarr::is_cube<decltype(t2)>::value);

	REQUIRE(noarr_test::type_is_simple(v));
	REQUIRE(noarr_test::type_is_simple(v2));
	REQUIRE(noarr_test::type_is_simple(t));
	REQUIRE(noarr_test::type_is_simple(t2));
}

TEST_CASE("Pipes sizes", "[sizes sizes]") {
	auto v = noarr::vector_t<'x', noarr::scalar<float>>();
	auto v2 = noarr::array_t<'y', 20000, noarr::vector_t<'x', noarr::scalar<float>>>();
	auto t = noarr::tuple_t<'t', noarr::array_t<'x', 10, noarr::scalar<float>>, noarr::vector_t<'y', noarr::scalar<int>>>();
	auto t2 = noarr::tuple_t<'t', noarr::array_t<'y', 20000, noarr::vector_t<'x', noarr::scalar<float>>>, noarr::vector_t<'a', noarr::array_t<'b', 20, noarr::scalar<int>>>>();

	auto v_sized = v ^ noarr::set_length<'x'>(20);

	REQUIRE((v_sized | noarr::get_length<'x'>()) == 20);
	REQUIRE((v2 | noarr::get_length<'y'>()) == 20000);
	REQUIRE((t ^ noarr::fix<'t'>(lit<0>) | noarr::get_length<'x'>()) == 10);
	REQUIRE((t2 ^ noarr::fix<'t'>(lit<0>) | noarr::get_length<'y'>()) == 20000);
	REQUIRE((t2 ^ noarr::fix<'t'>(lit<1>) | noarr::get_length<'b'>()) == 20);
}

TEST_CASE("Pipes resize", "[transform]") {
	noarr::vector_t<'x', noarr::scalar<float>> v;
	auto vs = v ^ noarr::set_length<'x'>(10);

	REQUIRE(noarr::is_cube<decltype(vs)>::value);

	REQUIRE(noarr_test::type_is_simple(vs));

	REQUIRE((vs | noarr::get_length<'x'>()) == 10);
}

TEST_CASE("Pipes resize 2", "[Resizing]") {
	noarr::array_t<'y', 20000, noarr::vector_t<'x', noarr::scalar<float>>> v2;
	auto vs2 = v2 ^ noarr::set_length<'x'>(20);

	REQUIRE(noarr::is_cube<decltype(vs2)>::value);

	REQUIRE(noarr_test::type_is_simple(vs2));

	REQUIRE(noarr_test::type_is_simple(vs2 ^ noarr::fix<'y', 'x'>(5, 5)));
	REQUIRE(noarr_test::type_is_simple(noarr::fix<'y', 'x'>(5, 5)));

	REQUIRE(!noarr::is_point<decltype(vs2)>::value);
	REQUIRE(noarr::is_point<decltype(vs2 ^ noarr::fix<'y', 'x'>(5, 5))>::value);

	REQUIRE((vs2 | noarr::get_length<'x'>()) == 20);
	REQUIRE((vs2 | noarr::get_length<'y'>()) == 20000);
}

TEST_CASE("Pipes resize 3", "[Resizing]") {
	noarr::array_t<'y', 20000, noarr::vector_t<'x', noarr::scalar<float>>> v2;
	auto vs3 = v2 ^ noarr::set_length<'x'>(lit<10>);

	REQUIRE(noarr::is_cube<decltype(vs3)>::value);

	REQUIRE(noarr_test::type_is_simple(vs3));

	REQUIRE((vs3 | noarr::get_length<'x'>()) == 10);
	REQUIRE((vs3 | noarr::get_length<'y'>()) == 20000);
}

TEST_CASE("Pipes resize 4", "[Resizing]") {
	volatile std::size_t l = 20;
	auto v2 = noarr::array_t<'y', 20000, noarr::vector_t<'x', noarr::scalar<float>>>();
	auto t = noarr::tuple_t<'t', noarr::array_t<'x', 10, noarr::scalar<float>>, noarr::vector_t<'x', noarr::scalar<int>>>();
	auto vs4 = v2 ^ noarr::set_length<'x'>(l);

	REQUIRE(noarr::is_cube<decltype(vs4)>::value);

	REQUIRE(noarr_test::type_is_simple(vs4));

	REQUIRE(!noarr::is_cube<decltype(t)>::value);

	REQUIRE(noarr_test::type_is_simple(t));
}

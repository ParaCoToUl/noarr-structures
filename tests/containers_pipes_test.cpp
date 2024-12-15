#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <noarr/structures_extended.hpp>

TEST_CASE("Pipes vector", "[resizing]")
{
	auto v = noarr::vector_t<'x', noarr::scalar<float>>();
	auto v2 = v ^ noarr::set_length<'x'>(10);

	REQUIRE((v2 | noarr::get_length<'x'>()) == 10);

	auto v3 = v  ^ noarr::set_length<'x'>(20);

	REQUIRE((v2 | noarr::get_length<'x'>()) == 10);
	REQUIRE((v3 | noarr::get_length<'x'>()) == 20);

	REQUIRE(!noarr::is_cube<decltype(v)>::value);
	REQUIRE( noarr::is_cube<decltype(v2)>::value);
	REQUIRE( noarr::is_cube<decltype(v3)>::value);

	REQUIRE((v2 | noarr::get_length<'x'>()) == 10);
	REQUIRE((v3 | noarr::get_length<'x'>()) == 20);
}

TEST_CASE("Pipes vector2", "[is_simple]")
{
	auto v = noarr::vector_t<'x', noarr::scalar<float>>();
	auto v2 = v ^ noarr::set_length<'x'>(10);

	REQUIRE(noarr_test::type_is_simple(v2));

	auto v3 = v ^ noarr::set_length<'x'>(20);

	REQUIRE(noarr_test::type_is_simple(v3));
}

TEST_CASE("Pipes do not affect bitwise or", "[is_simple]")
{
	auto num = 3 | 12;

	REQUIRE(num == 15);
}

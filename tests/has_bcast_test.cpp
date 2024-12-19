#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <cstddef>
#include <type_traits>

#include <noarr/structures_extended.hpp>

using namespace noarr;

// This test case tests the situation when the original length of the structure is not known
TEST_CASE("bcast test", "[has_test]") {
	constexpr auto testee = scalar<int>() ^ vector<'x'>(42) ^ vector<'y'>() ^ bcast<'z'>();

	STATIC_REQUIRE(!(testee | has_size()));
	STATIC_REQUIRE(testee | has_length<'x'>());
	STATIC_REQUIRE((testee | get_length<'x'>()) == 42);
	STATIC_REQUIRE(!(testee | has_length<'y'>()));
	STATIC_REQUIRE(!(testee | has_length<'z'>()));
	STATIC_REQUIRE(!(testee | has_offset()));

	STATIC_REQUIRE(testee ^ set_length<'y'>(2) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) | get_size()) == 42 * 2 * sizeof(int));
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) | get_length<'x'>()) == 42);
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) | get_length<'y'>()) == 2);
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) | has_length<'z'>()));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) | has_offset()));

	STATIC_REQUIRE(!(testee ^ set_length<'z'>(3) | has_size()));
	STATIC_REQUIRE(testee ^ set_length<'z'>(3) | has_length<'x'>());
	STATIC_REQUIRE((testee ^ set_length<'z'>(3) | get_length<'x'>()) == 42);
	STATIC_REQUIRE(!(testee ^ set_length<'z'>(3) | has_length<'y'>()));
	STATIC_REQUIRE(testee ^ set_length<'z'>(3) | has_length<'z'>());
	STATIC_REQUIRE((testee ^ set_length<'z'>(3) | get_length<'z'>()) == 3);
	STATIC_REQUIRE(!(testee ^ set_length<'z'>(3) | has_offset()));

	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | get_size()) == 42 * 2 * sizeof(int));
	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | has_length<'x'>());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | get_length<'x'>()) == 42);
	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | has_length<'y'>());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | get_length<'y'>()) == 2);
	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | has_length<'z'>());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | get_length<'z'>()) == 3);
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) | has_offset()));

	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ fix<'x'>(4) ^ fix<'y'>(1) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ fix<'x'>(4) ^ fix<'y'>(1) | get_size()) == 42 * 2 * sizeof(int));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ fix<'x'>(4) ^ fix<'y'>(1) | has_length<'x'>()));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ fix<'x'>(4) ^ fix<'y'>(1) | has_length<'y'>()));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ fix<'x'>(4) ^ fix<'y'>(1) | has_length<'z'>()));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | has_offset()));

	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | get_size()) == 42 * 2 * sizeof(int));
	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | has_length<'x'>());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | get_length<'x'>()) == 42);
	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | has_length<'y'>());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | get_length<'y'>()) == 2);
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | has_length<'z'>()));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'z'>(1) | has_offset()));

	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | get_size()) == 42 * 2 * sizeof(int));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | has_length<'x'>()));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | has_length<'y'>()));
	STATIC_REQUIRE(!(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | has_length<'z'>()));
	STATIC_REQUIRE(testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | has_offset());
	STATIC_REQUIRE((testee ^ set_length<'y'>(2) ^ set_length<'z'>(3) ^ fix<'x'>(4) ^ fix<'y'>(1) ^ fix<'z'>(0) | offset()) == 1 * 42 * sizeof(int) + 4 * sizeof(int));
}

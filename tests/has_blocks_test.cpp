#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <cstddef>
#include <type_traits>

#include <noarr/structures_extended.hpp>

using namespace noarr;

// This test case tests the situation when the original length of the structure is not known
TEST_CASE("into_blocks post test", "[has_test]") {
	constexpr auto testee = scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'X'>();

	STATIC_REQUIRE(!(testee | has_size()));
	STATIC_REQUIRE(!(testee | has_length<'x'>()));
	STATIC_REQUIRE(!(testee | has_offset()));

	STATIC_REQUIRE(testee ^ set_length<'x'>(10) ^ set_length<'X'>(2) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'x'>(10) ^ set_length<'X'>(2) | get_size()) == 10 * 2 * sizeof(int));

	STATIC_REQUIRE(testee ^ set_length<'x'>(10) | has_length<'x'>());
	STATIC_REQUIRE((testee ^ set_length<'x'>(10) | get_length<'x'>()) == 10);

	STATIC_REQUIRE(testee ^ set_length<'X'>(2) | has_length<'X'>());
	STATIC_REQUIRE((testee ^ set_length<'X'>(2) | get_length<'X'>()) == 2);

	STATIC_REQUIRE(!(testee ^ set_length<'x'>(10) | has_offset()));
	STATIC_REQUIRE(!(testee ^ set_length<'X'>(2) | has_offset()));

	STATIC_REQUIRE(!(testee ^ set_length<'x'>(10) ^ set_length<'X'>(2) | has_offset()));
	STATIC_REQUIRE(!(testee ^ set_length<'x'>(10) ^ set_length<'X'>(2) ^ fix<'x'>(4) | has_offset()));
	STATIC_REQUIRE(!(testee ^ set_length<'x'>(10) ^ set_length<'X'>(2) ^ fix<'X'>(1) | has_offset()));
	STATIC_REQUIRE(testee ^ set_length<'x'>(10) ^ set_length<'X'>(2) ^ fix<'x'>(4) ^ fix<'X'>(1) | has_offset());
	STATIC_REQUIRE((testee ^ set_length<'x'>(10) ^ set_length<'X'>(2) ^ fix<'x'>(4) ^ fix<'X'>(1) | offset()) == 1 * 10 * sizeof(int) + 4 * sizeof(int));
}

// This test case tests the situation when the original length of the structure is known
// - only one of the blocking dimensions is to be set
TEST_CASE("into_blocks pre test", "[has_test]") {
	constexpr auto testee = scalar<int>() ^ vector<'x'>(42) ^ into_blocks<'x', 'X'>();

	STATIC_REQUIRE((testee | has_size()));
	STATIC_REQUIRE((testee | get_size()) == 42 * sizeof(int));

	STATIC_REQUIRE(!(testee | has_length<'x'>()));

	STATIC_REQUIRE(!(testee | has_offset()));

	STATIC_REQUIRE(testee ^ set_length<'x'>(6) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'x'>(6) | get_size()) == 6 * 7 * sizeof(int));

	STATIC_REQUIRE(testee ^ set_length<'x'>(6) | has_length<'x'>());
	STATIC_REQUIRE((testee ^ set_length<'x'>(6) | get_length<'x'>()) == 6);
	STATIC_REQUIRE(testee ^ set_length<'x'>(6) | has_length<'X'>());
	STATIC_REQUIRE((testee ^ set_length<'x'>(6) | get_length<'X'>()) == 7);

	STATIC_REQUIRE(!(testee ^ set_length<'x'>(6) | has_offset()));

	STATIC_REQUIRE(!(testee ^ set_length<'x'>(6) ^ fix<'x'>(4) | has_offset()));
	STATIC_REQUIRE(!(testee ^ set_length<'x'>(6) ^ fix<'X'>(1) | has_offset()));
	STATIC_REQUIRE(testee ^ set_length<'x'>(6) ^ fix<'x'>(4) ^ fix<'X'>(1) | has_offset());
	STATIC_REQUIRE((testee ^ set_length<'x'>(6) ^ fix<'x'>(4) ^ fix<'X'>(1) | offset()) == 1 * 6 * sizeof(int) + 4 * sizeof(int));

	STATIC_REQUIRE(testee ^ set_length<'X'>(7) | has_size());
	STATIC_REQUIRE((testee ^ set_length<'X'>(7) | get_size()) == 6 * 7 * sizeof(int));

	STATIC_REQUIRE(testee ^ set_length<'X'>(7) | has_length<'x'>());
	STATIC_REQUIRE((testee ^ set_length<'X'>(7) | get_length<'x'>()) == 6);
	STATIC_REQUIRE(testee ^ set_length<'X'>(7) | has_length<'X'>());
	STATIC_REQUIRE((testee ^ set_length<'X'>(7) | get_length<'X'>()) == 7);

	STATIC_REQUIRE(!(testee ^ set_length<'X'>(7) | has_offset()));

	STATIC_REQUIRE(!(testee ^ set_length<'X'>(7) ^ fix<'x'>(4) | has_offset()));
	STATIC_REQUIRE(!(testee ^ set_length<'X'>(7) ^ fix<'X'>(1) | has_offset()));
	STATIC_REQUIRE(testee ^ set_length<'X'>(7) ^ fix<'x'>(4) ^ fix<'X'>(1) | has_offset());
	STATIC_REQUIRE((testee ^ set_length<'X'>(7) ^ fix<'x'>(4) ^ fix<'X'>(1) | offset()) == 1 * 6 * sizeof(int) + 4 * sizeof(int));
}

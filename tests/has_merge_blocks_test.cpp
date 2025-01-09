#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <noarr/structures_extended.hpp>

using namespace noarr;

// This test case tests the situation when the minor length of the structure is not known
TEST_CASE("merge_blocks minor test", "[has_test]") {
	// x is minor, y is major
	constexpr auto testee = scalar<int>() ^ vector<'x'>() ^ vector<'y'>(42) ^ merge_blocks<'y', 'x', 'm'>();

	STATIC_REQUIRE(!(testee | has_size())); // not known
	STATIC_REQUIRE(!(testee | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee | has_length<'m'>())); // not known
	STATIC_REQUIRE(!(testee | has_offset())); // not known

	STATIC_REQUIRE(testee ^ set_length<'m'>(420) | has_size()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(testee ^ set_length<'m'>(420) | has_length<'m'>()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) | get_length<'m'>()) == 420);
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) | has_offset())); // not known

	STATIC_REQUIRE(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_size()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'m'>())); // already fixed
	STATIC_REQUIRE(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_offset()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | offset()) == 87 * sizeof(int));

	// x is major, y is minor
	constexpr auto testee2 = scalar<int>() ^ vector<'x'>(42) ^ vector<'y'>() ^ merge_blocks<'x', 'y', 'm'>();

	STATIC_REQUIRE(!(testee2 | has_size())); // not known
	STATIC_REQUIRE(!(testee2 | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee2 | has_length<'m'>())); // not known
	STATIC_REQUIRE(!(testee2 | has_offset())); // not known

	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) | has_size()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) | has_length<'m'>()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) | get_length<'m'>()) == 420);
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) | has_offset())); // not known

	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_size()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'m'>())); // already fixed
	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_offset()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(1) | offset()) == 42 * sizeof(int));
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(10) | offset()) == 1 * sizeof(int));
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | offset()) == (87 / 10 + 87 % 10 * 42) * sizeof(int));
}

// This test case tests the situation when the major length of the structure is not known
TEST_CASE("merge_blocks major test", "[has_test]") {
	// x is minor, y is major
	constexpr auto testee = scalar<int>() ^ vector<'x'>() ^ vector<'y'>(42) ^ merge_blocks<'y', 'x', 'm'>();

	STATIC_REQUIRE(!(testee | has_size())); // not known
	STATIC_REQUIRE(!(testee | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee | has_length<'m'>())); // not known
	STATIC_REQUIRE(!(testee | has_offset())); // not known

	STATIC_REQUIRE(testee ^ set_length<'m'>(420) | has_size()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(testee ^ set_length<'m'>(420) | has_length<'m'>()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) | get_length<'m'>()) == 420);
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) | has_offset())); // not known

	STATIC_REQUIRE(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_size()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'m'>())); // already fixed
	STATIC_REQUIRE(testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_offset()); // known
	STATIC_REQUIRE((testee ^ set_length<'m'>(420) ^ fix<'m'>(87) | offset()) == 87 * sizeof(int));

	// x is major, y is minor
	constexpr auto testee2 = scalar<int>() ^ vector<'x'>(42) ^ vector<'y'>() ^ merge_blocks<'x', 'y', 'm'>();

	STATIC_REQUIRE(!(testee2 | has_size())); // not known
	STATIC_REQUIRE(!(testee2 | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee2 | has_length<'m'>())); // not known
	STATIC_REQUIRE(!(testee2 | has_offset())); // not known

	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) | has_size()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) | has_length<'m'>()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) | get_length<'m'>()) == 420);
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) | has_offset())); // not known

	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_size()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_length<'m'>())); // already fixed
	STATIC_REQUIRE(testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | has_offset()); // known
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(1) | offset()) == 42 * sizeof(int));
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(10) | offset()) == 1 * sizeof(int));
	STATIC_REQUIRE((testee2 ^ set_length<'m'>(420) ^ fix<'m'>(87) | offset()) == (87 / 10 + 87 % 10 * 42) * sizeof(int));
}

// This test case tests the situation when both the major and minor lengths of the structure are known
TEST_CASE("merge_blocks both test", "[has_test]") {
	// x is minor, y is major
	constexpr auto testee = scalar<int>() ^ vector<'x'>(10) ^ vector<'y'>(42) ^ merge_blocks<'y', 'x', 'm'>();

	STATIC_REQUIRE((testee | has_size())); // known
	STATIC_REQUIRE((testee | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee | has_length<'y'>())); // not visible
	STATIC_REQUIRE(testee | has_length<'m'>()); // known
	STATIC_REQUIRE((testee | get_length<'m'>()) == 420);
	STATIC_REQUIRE(!(testee | has_offset())); // not known

	STATIC_REQUIRE(testee ^ fix<'m'>(87) | has_size()); // known
	STATIC_REQUIRE((testee ^ fix<'m'>(87) | get_size()) == 10 * 42 * sizeof(int));
	STATIC_REQUIRE(!(testee ^ fix<'m'>(87) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee ^ fix<'m'>(87) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee ^ fix<'m'>(87) | has_length<'m'>())); // already fixed
	STATIC_REQUIRE(testee ^ fix<'m'>(87) | has_offset()); // known
	STATIC_REQUIRE((testee ^ fix<'m'>(87) | offset()) == 87 * sizeof(int));

	// x is major, y is minor
	constexpr auto testee2 = scalar<int>() ^ vector<'x'>(42) ^ vector<'y'>(10) ^ merge_blocks<'x', 'y', 'm'>();

	STATIC_REQUIRE((testee2 | has_size())); // known
	STATIC_REQUIRE((testee2 | get_size()) == 42 * 10 * sizeof(int));
	STATIC_REQUIRE(!(testee2 | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 | has_length<'y'>())); // not visible
	STATIC_REQUIRE(testee2 | has_length<'m'>()); // known
	STATIC_REQUIRE((testee2 | get_length<'m'>()) == 420);
	STATIC_REQUIRE(!(testee2 | has_offset())); // not known

	STATIC_REQUIRE(testee2 ^ fix<'m'>(87) | has_size()); // known
	STATIC_REQUIRE((testee2 ^ fix<'m'>(87) | get_size()) == 42 * 10 * sizeof(int));
	STATIC_REQUIRE(!(testee2 ^ fix<'m'>(87) | has_length<'x'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ fix<'m'>(87) | has_length<'y'>())); // not visible
	STATIC_REQUIRE(!(testee2 ^ fix<'m'>(87) | has_length<'m'>())); // already fixed
	STATIC_REQUIRE(testee2 ^ fix<'m'>(87) | has_offset()); // known
	STATIC_REQUIRE((testee2 ^ fix<'m'>(1) | offset()) == 42 * sizeof(int));
	STATIC_REQUIRE((testee2 ^ fix<'m'>(10) | offset()) == 1 * sizeof(int));
	STATIC_REQUIRE((testee2 ^ fix<'m'>(87) | offset()) == (87 / 10 + 87 % 10 * 42) * sizeof(int));
}

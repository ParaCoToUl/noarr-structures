#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <cstddef>

#include <noarr/structures_extended.hpp>

using namespace noarr;

TEST_CASE("scalar test", "[has_test]") {
	constexpr auto testee = scalar<int>();

	STATIC_REQUIRE(testee | has_size());
	STATIC_REQUIRE((testee | get_size()) == sizeof(int));

	STATIC_REQUIRE(!(testee | has_length<'x'>()));

	STATIC_REQUIRE((testee | has_offset()));
	STATIC_REQUIRE((testee | offset()) == 0);
}

TEST_CASE("array test", "[has_test]") {
	constexpr auto testee = scalar<int>() ^ array<'x', 10>();

	STATIC_REQUIRE(testee | has_size());
	STATIC_REQUIRE((testee | get_size()) == 10 * sizeof(int));

	STATIC_REQUIRE(testee | has_length<'x'>());
	STATIC_REQUIRE((testee | get_length<'x'>()) == 10);
	STATIC_REQUIRE(!(testee | has_length<'y'>()));

	STATIC_REQUIRE(!(testee | has_offset()));

	constexpr std::size_t idx = 4;
	constexpr auto testee2 = testee ^ fix<'x'>(idx);

	STATIC_REQUIRE(testee2 | has_size());
	STATIC_REQUIRE((testee2 | get_size()) == 10 * sizeof(int));

	STATIC_REQUIRE(!(testee2 | has_length<'x'>()));

	STATIC_REQUIRE(testee2 | has_offset());
	STATIC_REQUIRE((testee2 | offset()) == 4 * sizeof(int));

	constexpr auto testee3 = testee ^ fix<'y'>(4);

	STATIC_REQUIRE(testee3 | has_size());
	STATIC_REQUIRE((testee3 | get_size()) == 10 * sizeof(int));

	STATIC_REQUIRE(testee3 | has_length<'x'>());
	STATIC_REQUIRE((testee3 | get_length<'x'>()) == 10);
	STATIC_REQUIRE(!(testee3 | has_length<'y'>()));

	STATIC_REQUIRE(!(testee3 | has_offset()));

	constexpr auto testee4 = testee2 ^ fix<'y'>(4);

	STATIC_REQUIRE(testee4 | has_size());
	STATIC_REQUIRE((testee4 | get_size()) == 10 * sizeof(int));

	STATIC_REQUIRE(!(testee4 | has_length<'x'>()));
	STATIC_REQUIRE(!(testee4 | has_length<'y'>()));

	STATIC_REQUIRE(testee4 | has_offset());
	STATIC_REQUIRE((testee4 | offset()) == 4 * sizeof(int));
}

TEST_CASE("vector test", "[has_test]") {
	constexpr auto testee = scalar<int>() ^ vector<'x'>();

	STATIC_REQUIRE(!(testee | has_size()));
	STATIC_REQUIRE(!(testee | has_length<'x'>()));
	STATIC_REQUIRE(!(testee | has_offset()));

	constexpr auto testee2 = testee ^ set_length<'x'>(10);

	STATIC_REQUIRE(testee2 | has_size());

	STATIC_REQUIRE(testee2 | has_length<'x'>());
	STATIC_REQUIRE((testee2 | get_length<'x'>()) == 10);

	STATIC_REQUIRE(!(testee2 | has_offset()));
	STATIC_REQUIRE(testee2 ^ fix<'x'>(4) | has_offset());

	constexpr auto testee3 = testee2 ^ fix<'x'>(4);

	STATIC_REQUIRE(testee3 | has_size());

	STATIC_REQUIRE(!(testee3 | has_length<'x'>()));

	STATIC_REQUIRE(testee3 | has_offset());
	STATIC_REQUIRE((testee3 | offset()) == 4 * sizeof(int));


	constexpr auto testee4 = testee ^ set_length<'y'>(10);

	STATIC_REQUIRE(!(testee4 | has_size()));
	STATIC_REQUIRE(!(testee4 | has_length<'x'>()));
	STATIC_REQUIRE(!(testee4 | has_length<'y'>()));
	STATIC_REQUIRE(!(testee4 | has_offset()));
}

TEST_CASE("tuple test", "[has_test]") {
	constexpr auto testee = pack(scalar<int>(), scalar<float>()) ^ tuple<'x'>();

	STATIC_REQUIRE((testee | has_size()));
	STATIC_REQUIRE((testee | get_size()) == sizeof(int) + sizeof(float));

	STATIC_REQUIRE((testee | has_length<'x'>()));
	STATIC_REQUIRE((testee | get_length<'x'>()) == 2);
	STATIC_REQUIRE(!(testee | has_length<'y'>()));

	STATIC_REQUIRE(!(testee | has_offset()));

	constexpr auto idx = lit<1>;
	constexpr auto testee2 = testee ^ fix<'x'>(idx);

	STATIC_REQUIRE((testee2 | has_offset()));
	STATIC_REQUIRE((testee2 | offset()) == sizeof(int));
}

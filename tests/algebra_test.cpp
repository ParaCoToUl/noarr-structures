#include <catch2/catch_test_macros.hpp>
#include <type_traits>

#include <noarr/structures_extended.hpp>

using namespace noarr;

TEST_CASE("Simple array algebra", "[algebra]") {
	auto testee = scalar<int>() ^ array<'x', 10>();
	auto reference = array<'x', 10, scalar<int>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Simple vector algebra", "[algebra]") {
	auto testee = scalar<int>() ^ vector<'x'>();
	auto reference = vector<'x', scalar<int>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Sized vector algebra", "[algebra]") {
	auto testee = scalar<int>() ^ (vector<'x'>() ^ set_length<'x'>(10));
	auto reference = vector<'x', scalar<int>>() ^ set_length<'x'>(10);

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
	REQUIRE((testee | get_length<'x'>()) == (reference | get_length<'x'>()));
}

TEST_CASE("Algebra preseves bitwise xor", "[algebra]") {
	auto num = 3 ^ 12;

	REQUIRE(num == 15);
}

TEST_CASE("Composite array algebra", "[algebra]") {
	auto testee = scalar<int>() ^ array<'x', 10>() ^ array<'y', 20>();
	auto reference = array<'y', 20, array<'x', 10, scalar<int>>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Composite vector algebra", "[algebra]") {
	auto testee = scalar<int>() ^ vector<'x'>() ^ vector<'y'>();
	auto reference = vector<'y', vector<'x', scalar<int>>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Sized vector test", "[algebra shortcuts]") {
	auto testee = scalar<int>() ^ sized_vector<'x'>(20) ^ sized_vector<'y'>(30);
	using reference_t = set_length_t<'y', vector<'y', set_length_t<'x', vector<'x', scalar<int>>, std::size_t>>, std::size_t>;

	REQUIRE(std::is_same<decltype(testee), reference_t>::value);
	REQUIRE((testee | get_size()) == (20 * 30 * sizeof(int)));
}

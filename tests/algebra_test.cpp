#include <catch2/catch_test_macros.hpp>
#include <type_traits>

#include <noarr/structures_extended.hpp>

using namespace noarr;

TEST_CASE("Simple array algebra", "[algebra]") {
	auto testee = scalar<int>() ^ array<'x', 10>();
	auto reference = array_t<'x', 10, scalar<int>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Simple vector algebra", "[algebra]") {
	auto testee = scalar<int>() ^ vector<'x'>();
	auto reference = vector_t<'x', scalar<int>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Sized vector algebra", "[algebra]") {
	auto testee = scalar<int>() ^ (vector<'x'>() ^ set_length<'x'>(10));
	auto testee2 = (scalar<int>() ^ vector<'x'>()) ^ set_length<'x'>(10);

	auto reference = vector_t<'x', scalar<int>>() ^ set_length<'x'>(10);

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
	REQUIRE(std::is_same<decltype(testee2), decltype(reference)>::value);

	REQUIRE((testee | get_length<'x'>()) == (reference | get_length<'x'>()));
	REQUIRE((testee2 | get_length<'x'>()) == (reference | get_length<'x'>()));
}

TEST_CASE("Simple tuple algebra", "[algebra]") {
	auto testee = pack(scalar<int>(), scalar<float>()) ^ tuple<'x'>();
	auto reference = tuple_t<'x', scalar<int>, scalar<float>>();

	auto testee2 = scalar<int>() ^ tuple<'x'>();
	auto reference2 = tuple_t<'x', scalar<int>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
	REQUIRE(std::is_same<decltype(testee2), decltype(reference2)>::value);
}

TEST_CASE("Algebra preseves bitwise xor", "[algebra]") {
	auto num = 3 ^ 12;

	REQUIRE(num == 15);
}

TEST_CASE("Composite array algebra", "[algebra]") {
	auto testee = scalar<int>() ^ array<'x', 10>() ^ array<'y', 20>();
	auto testee2 = scalar<int>() ^ (array<'x', 10>() ^ array<'y', 20>());

	auto reference = array_t<'y', 20, array_t<'x', 10, scalar<int>>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
	REQUIRE(std::is_same<decltype(testee2), decltype(reference)>::value);
}

TEST_CASE("Composite vector algebra", "[algebra]") {
	auto testee = scalar<int>() ^ vector<'x'>() ^ vector<'y'>();
	auto testee2 = scalar<int>() ^ (vector<'x'>() ^ vector<'y'>());

	auto reference = vector_t<'y', vector_t<'x', scalar<int>>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
	REQUIRE(std::is_same<decltype(testee2), decltype(reference)>::value);
}

TEST_CASE("Composite tuple-array algebra", "[algebra]") {
	auto testee = scalar<int>() ^ (pack(array<'y', 10>(), array<'z', 20>()) ^ tuple<'x'>());
	auto testee2 = scalar<int>() ^ pack(array<'y', 10>(), array<'z', 20>()) ^ tuple<'x'>();

	auto reference = pack(scalar<int>() ^ array<'y', 10>(), scalar<int>() ^ array<'z', 20>()) ^ tuple<'x'>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
	REQUIRE(std::is_same<decltype(testee2), decltype(reference)>::value);
}

TEST_CASE("Composite tuple-tuple algebra", "[algebra]") {
	auto testee = pack(scalar<int>(), scalar<float>()) ^ pack(tuple<'y'>(), tuple<'z'>()) ^ tuple<'x'>();
	auto testee2 = pack(scalar<int>(), scalar<float>()) ^ (pack(tuple<'y'>(), tuple<'z'>()) ^ tuple<'x'>());


	auto reference = pack(pack(scalar<int>(), scalar<float>()) ^ tuple<'y'>(), pack(scalar<int>(), scalar<float>()) ^ tuple<'z'>()) ^ tuple<'x'>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
	REQUIRE(std::is_same<decltype(testee2), decltype(reference)>::value);
}

TEST_CASE("Sized vector test", "[algebra shortcuts]") {
	auto testee = scalar<int>() ^ sized_vector<'x'>(20) ^ sized_vector<'y'>(30);
	auto testee2 = scalar<int>() ^ (sized_vector<'x'>(20) ^ sized_vector<'y'>(30));

	using reference_t = set_length_t<'y', vector_t<'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, std::size_t>;

	REQUIRE(std::is_same<decltype(testee), reference_t>::value);
	REQUIRE((testee | get_size()) == (20 * 30 * sizeof(int)));

	REQUIRE(std::is_same<decltype(testee2), reference_t>::value);
	REQUIRE((testee2 | get_size()) == (20 * 30 * sizeof(int)));
}

TEST_CASE("Simple to_each algebra", "[algebra]") {
	auto testee = pack(scalar<int>(), scalar<float>()) ^ to_each(vector<'x'>());
	auto reference = pack<vector_t<'x', scalar<int>>, vector_t<'x', scalar<float>>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Composite to_each algebra", "[algebra]") {
	auto testee = pack(scalar<int>(), scalar<float>()) ^ pack(to_each(vector<'x'>()), to_each(vector<'y'>()));
	auto reference = pack<pack<vector_t<'x', scalar<int>>, vector_t<'x', scalar<float>>>, pack<vector_t<'y', scalar<int>>, vector_t<'y', scalar<float>>>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

TEST_CASE("Composite to_each algebra with just one vector", "[algebra]") {
	auto testee = pack(scalar<int>(), scalar<float>()) ^ pack(to_each(vector<'x'>()));
	auto reference = pack<pack<vector_t<'x', scalar<int>>, vector_t<'x', scalar<float>>>>();

	REQUIRE(std::is_same<decltype(testee), decltype(reference)>::value);
}

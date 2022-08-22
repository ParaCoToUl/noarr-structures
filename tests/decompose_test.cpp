#include <catch2/catch.hpp>

#include "noarr/structures.hpp"
#include "noarr/structures/decompose.hpp"

TEST_CASE("Decompose", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::decompose<'x', 'b', 'a'>(16);
	
	REQUIRE((m | noarr::offset<'a', 'y', 'b'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Decompose reused as minor", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::decompose<'x', 'X', 'x'>(16);
	
	REQUIRE((m | noarr::offset<'x', 'y', 'X'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Decompose reused as major", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::decompose<'x', 'x', 'X'>(16);
	
	REQUIRE((m | noarr::offset<'X', 'y', 'x'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Decompose set length", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::decompose<'x', 'b', 'a'>(16)
		^ noarr::set_length<'b'>(10'000/16);
	
	REQUIRE((m | noarr::offset<'a', 'y', 'b'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Decompose set length reused as minor", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::decompose<'x', 'X', 'x'>(16)
		^ noarr::set_length<'X'>(10'000/16);
	
	REQUIRE((m | noarr::offset<'x', 'y', 'X'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Decompose set length reused as major", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::decompose<'x', 'x', 'X'>(16)
		^ noarr::set_length<'x'>(10'000/16);
	
	REQUIRE((m | noarr::offset<'X', 'y', 'x'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

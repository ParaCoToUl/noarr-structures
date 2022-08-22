#include <catch2/catch.hpp>

#include "noarr/structures.hpp"
#include "noarr/structures/compose.hpp"

TEST_CASE("Compose", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 500>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::compose<'x', 'z', 'w'>();
	
	REQUIRE((m | noarr::offset<'w', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Compose reused minor", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 500>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::compose<'x', 'z', 'z'>();
	
	REQUIRE((m | noarr::offset<'z', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Compose reused major", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 500>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::compose<'x', 'z', 'x'>();
	
	REQUIRE((m | noarr::offset<'x', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Compose set length", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::compose<'x', 'z', 'w'>()
		^ noarr::set_length<'w'>(500*700L);
	
	REQUIRE((m | noarr::offset<'w', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Compose set length reused minor", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::compose<'x', 'z', 'z'>()
		^ noarr::set_length<'z'>(500*700L);
	
	REQUIRE((m | noarr::offset<'z', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Compose set length reused major", "[decompose]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::compose<'x', 'z', 'x'>()
		^ noarr::set_length<'x'>(500*700L);
	
	REQUIRE((m | noarr::offset<'x', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

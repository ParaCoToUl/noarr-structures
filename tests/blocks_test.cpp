#include <catch2/catch.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/structs/blocks.hpp>
#include <noarr/structures/extra/shortcuts.hpp>

TEST_CASE("Split", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks<'x', 'b', 'a'>(16);
	
	REQUIRE((m | noarr::offset<'a', 'y', 'b'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split reused as minor", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks<'x', 'X', 'x'>(16);
	
	REQUIRE((m | noarr::offset<'x', 'y', 'X'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split reused as major", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks<'x', 'x', 'X'>(16);
	
	REQUIRE((m | noarr::offset<'X', 'y', 'x'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split set length", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks<'x', 'b', 'a'>(16)
		^ noarr::set_length<'b'>(10'000/16);
	
	REQUIRE((m | noarr::offset<'a', 'y', 'b'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split set length reused as minor", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks<'x', 'X', 'x'>(16)
		^ noarr::set_length<'X'>(10'000/16);
	
	REQUIRE((m | noarr::offset<'x', 'y', 'X'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split set length reused as major", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks<'x', 'x', 'X'>(16)
		^ noarr::set_length<'x'>(10'000/16);
	
	REQUIRE((m | noarr::offset<'X', 'y', 'x'>(10, 3333, 500)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Merge", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 500>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::merge_blocks<'x', 'z', 'w'>();
	
	REQUIRE((m | noarr::offset<'w', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Merge reused minor", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 500>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::merge_blocks<'x', 'z', 'z'>();
	
	REQUIRE((m | noarr::offset<'z', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Merge reused major", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 500>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::merge_blocks<'x', 'z', 'x'>();
	
	REQUIRE((m | noarr::offset<'x', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Merge set length", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::merge_blocks<'x', 'z', 'w'>()
		^ noarr::set_length<'w'>(500*700L);
	
	REQUIRE((m | noarr::offset<'w', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Merge set length reused minor", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::merge_blocks<'x', 'z', 'z'>()
		^ noarr::set_length<'z'>(500*700L);
	
	REQUIRE((m | noarr::offset<'z', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Merge set length reused major", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'z', 700>()
		^ noarr::merge_blocks<'x', 'z', 'x'>()
		^ noarr::set_length<'x'>(500*700L);
	
	REQUIRE((m | noarr::offset<'x', 'y'>(1100, 123)) == (1100/700 + 123*500 + 1100%700*500L*600L) * sizeof(float));
}

TEST_CASE("Split get length, set from inside, array", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::into_blocks<'x', 'b', 'a'>(16);

	REQUIRE((m | noarr::get_length<'a'>()) == 16);
	REQUIRE((m | noarr::get_length<'b'>()) == 10'000/16);
	REQUIRE((m | noarr::get_size()) == 10'000 * sizeof(float));
}

TEST_CASE("Split get length, set from inside, vector", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::set_length<'x'>(10'000)
		^ noarr::into_blocks<'x', 'b', 'a'>(16);

	REQUIRE((m | noarr::get_length<'a'>()) == 16);
	REQUIRE((m | noarr::get_length<'b'>()) == 10'000/16);
	REQUIRE((m | noarr::get_size()) == 10'000 * sizeof(float));
}

TEST_CASE("Split get length, set from outside", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::into_blocks<'x', 'b', 'a'>(16)
		^ noarr::set_length<'b'>(625);

	REQUIRE((m | noarr::get_length<'a'>()) == 16);
	REQUIRE((m | noarr::get_length<'b'>()) == 625);
	REQUIRE((m | noarr::get_size()) == 625*16 * sizeof(float));
}

TEST_CASE("Split get length, set from outside, reversed", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::into_blocks<'x', 'b', 'a'>()
		^ noarr::set_length<'b'>(625)
		^ noarr::set_length<'a'>(16);

	REQUIRE((m | noarr::get_length<'a'>()) == 16);
	REQUIRE((m | noarr::get_length<'b'>()) == 625);
	REQUIRE((m | noarr::get_size()) == 625*16 * sizeof(float));
}

TEST_CASE("Merge get length, set from inside, array", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'a', 16>()
		^ noarr::array<'b', 625>()
		^ noarr::merge_blocks<'b', 'a', 'x'>();

	REQUIRE((m | noarr::get_length<'x'>()) == 10'000);
	REQUIRE((m | noarr::get_size()) == 10'000 * sizeof(float));
}

TEST_CASE("Merge get length, set from inside, vector", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'a'>()
		^ noarr::vector<'b'>()
		^ noarr::set_length<'a'>(16)
		^ noarr::set_length<'b'>(625)
		^ noarr::merge_blocks<'b', 'a', 'x'>();

	REQUIRE((m | noarr::get_length<'x'>()) == 10'000);
	REQUIRE((m | noarr::get_size()) == 10'000 * sizeof(float));
}

TEST_CASE("Merge get length, set from outside", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'a'>()
		^ noarr::vector<'b'>()
		^ noarr::set_length<'a'>(16)
		^ noarr::merge_blocks<'b', 'a', 'x'>()
		^ noarr::set_length<'x'>(10'000);

	REQUIRE((m | noarr::get_length<'x'>()) == 10'000);
	REQUIRE((m | noarr::get_size()) == 10'000 * sizeof(float));
}

#include <noarr_test/macros.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/structs/blocks.hpp>
#include <noarr/structures/extra/shortcuts.hpp>


TEST_CASE("Split dynamic", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'b', 'a', '_'>(16);

	REQUIRE((m | noarr::get_length<'_'>(noarr::idx<'a', 'y', 'b'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'a', 'y', 'b', '_'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic reused as flag", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'b', 'a', 'x'>(16);

	REQUIRE((m | noarr::get_length<'x'>(noarr::idx<'a', 'y', 'b'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'a', 'y', 'b', 'x'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic reused as minor", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'X', 'x', '_'>(16);

	REQUIRE((m | noarr::get_length<'_'>(noarr::idx<'x', 'y', 'X'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'x', 'y', 'X', '_'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic reused as major", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::array<'x', 10'000>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'x', 'X', '_'>(16);

	REQUIRE((m | noarr::get_length<'_'>(noarr::idx<'X', 'y', 'x'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'X', 'y', 'x', '_'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic set length", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'b', 'a', '_'>(16)
		^ noarr::set_length<'b'>(10'000/16);

	REQUIRE((m | noarr::get_length<'_'>(noarr::idx<'a', 'y', 'b'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'a', 'y', 'b', '_'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic set length reused as flag", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'b', 'a', 'x'>(16)
		^ noarr::set_length<'b'>(10'000/16);

	REQUIRE((m | noarr::get_length<'x'>(noarr::idx<'a', 'y', 'b'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'a', 'y', 'b', 'x'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic set length reused as minor", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'X', 'x', '_'>(16)
		^ noarr::set_length<'X'>(10'000/16);

	REQUIRE((m | noarr::get_length<'_'>(noarr::idx<'x', 'y', 'X'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'x', 'y', 'X', '_'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic set length reused as major", "[blocks]") {
	auto m = noarr::scalar<float>()
		^ noarr::vector<'x'>()
		^ noarr::array<'y', 20'000>()
		^ noarr::into_blocks_dynamic<'x', 'x', 'X', '_'>(16)
		^ noarr::set_length<'x'>(10'000/16);

	REQUIRE((m | noarr::get_length<'_'>(noarr::idx<'X', 'y', 'x'>(10, 3333, 500))) == 1);
	REQUIRE((m | noarr::offset<'X', 'y', 'x', '_'>(10, 3333, 500, 0)) == (10 + 500*16 + 3333*10'000L) * sizeof(float));
}

TEST_CASE("Split dynamic remainder", "[blocks]") {
	for(std::size_t xlen = 0; xlen < 20; xlen++) {
		auto m = noarr::scalar<float>()
			^ noarr::vector<'x'>(xlen)
			^ noarr::into_blocks_dynamic<'x', 'b', 'a', '_'>(4);

		std::size_t block_size = m | noarr::get_length<'a'>();
		std::size_t block_cnt = m | noarr::get_length<'b'>();
		std::size_t addressable = block_size * block_cnt;

		// we want to see the block size we asked for
		REQUIRE(block_size == 4);
		// there should be enough blocks to cover all elems
		REQUIRE(addressable >= xlen);
		// but there should be a block or more remaining
		REQUIRE(addressable - xlen < block_size);

		for(std::size_t x = 0; x < addressable; x++) {
			// block index
			std::size_t b = x / 4;
			// index within block
			std::size_t a = x % 4;

			auto state = noarr::idx<'a', 'b'>(a, b);

			if(x < xlen) {
				// should be valid elem
				REQUIRE((m | noarr::get_length<'_'>(state)) == 1);
				REQUIRE((m | noarr::offset<'b', 'a', '_'>(b, a, 0)) == x * sizeof(float));
			} else {
				// should not be present
				REQUIRE((m | noarr::get_length<'_'>(state)) == 0);
			}
		}
	}
}

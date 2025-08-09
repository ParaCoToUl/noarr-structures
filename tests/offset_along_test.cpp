#include <noarr_test/macros.hpp>

#include <cstddef>

#include <type_traits>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/introspection/offset_along.hpp>

using namespace noarr;


TEST_CASE("bcast_t", "[offset_along]") {
	STATIC_REQUIRE(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 0 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
										state<state_item<index_in<'x'>, std::size_t>>(5)) == 0 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
										state<state_item<index_in<'x'>, std::size_t>>(5)) == 0 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
										state<state_item<length_in<'y'>, std::size_t>>(42)) == 0 * sizeof(int));
	STATIC_REQUIRE(
		HasOffsetAlong<bcast_t<'x', scalar<int>>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 0 * sizeof(int));
}

TEST_CASE("vector_t", "[offset_along]") {
	STATIC_REQUIRE(HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 5 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		HasOffsetAlong<vector_t<'x', scalar<int>>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 5 * sizeof(int));

	STATIC_REQUIRE(
		HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
										state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(42, 5)) == 5 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
									state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
						state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'y'>(
					scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
					state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(42, 42, 5)) ==
				5 * sizeof(int));
	STATIC_REQUIRE(
		HasOffsetAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
						state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(
					scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
					state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(5, 42, 3)) == 3 * 42 * sizeof(int));
}

TEST_CASE("tuple_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, lit_t<0>>>>);
	STATIC_REQUIRE(offset_along<'x'>(pack(scalar<int>(), scalar<int>()) ^ tuple<'x'>(),
										state<state_item<index_in<'x'>, lit_t<0>>>()) == 0 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, lit_t<1>>>>);
	STATIC_REQUIRE(offset_along<'x'>(pack(scalar<int>(), scalar<int>()) ^ tuple<'x'>(),
										state<state_item<index_in<'x'>, lit_t<1>>>()) == 1 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, lit_t<2>>>>);
}

TEST_CASE("scalar", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<scalar<int>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<scalar<int>, 'x',
									state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasOffsetAlong<scalar<int>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("fix_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(
		!HasOffsetAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<fix_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
									state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<fix_t<'x', vector_t<'y', scalar<int>>, std::size_t>, 'y',
									state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'y'>() ^ fix<'x'>(3),
										state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 4)) == 4 * sizeof(int));
}

TEST_CASE("set_length_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(HasOffsetAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(3), state<state_item<index_in<'x'>, std::size_t>>(2)) == 2 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ set_length<'x'>(6), state<>()) == 0 * sizeof(int));
}

TEST_CASE("rename_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<rename_t<scalar<int>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ rename<'x', 'y'>(),
										state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3)) == 3 * sizeof(int));
}

TEST_CASE("join_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<length_in<'z'>, std::size_t>, state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ join<'x', 'y', 'z'>(),
										state<state_item<length_in<'z'>, std::size_t>, state_item<index_in<'z'>, std::size_t>>(5, 3)) == 3 * sizeof(int) + 3 * 5 * sizeof(int));
}

TEST_CASE("shift_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ shift<'x'>(3), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(6, 2)) == 5 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<shift_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ shift<'x'>(3), state<>()) == 0 * sizeof(int));
}

TEST_CASE("slice_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ slice<'x'>(3, 3), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(6, 2)) == 5 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<slice_t<'x', bcast_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ slice<'x'>(3, 3), state<state_item<index_in<'x'>, std::size_t>>(2)) == 0 * sizeof(int));
}

TEST_CASE("span_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ span<'x'>(3, 6), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(6, 2)) == 5 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<span_t<'x', bcast_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ span<'x'>(3, 6), state<state_item<index_in<'x'>, std::size_t>>(2)) == 0 * sizeof(int));
}

TEST_CASE("step_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(HasOffsetAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'x'>(3, 2), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 4)) == (4 * 2 + 3) * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<step_t<'x', bcast_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ step<'x'>(3, 2), state<state_item<index_in<'x'>, std::size_t>>(4)) == 0 * sizeof(int));
}

TEST_CASE("reverse_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<>>);
	STATIC_REQUIRE(HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>(42, 5)) == 36 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<reverse_t<'x', bcast_t<'x', scalar<int>>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ reverse<'x'>(), state<state_item<index_in<'x'>, std::size_t>>(5)) == 0 * sizeof(int));
}

TEST_CASE("into_blocks_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(
		!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

								STATIC_REQUIRE(HasOffsetAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
									state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 7, 3, 2)) == 3 * sizeof(int));
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 3 * sizeof(int));
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(7, 3, 2)) == 3 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
									state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 7, 3, 2)) == 2 * 5 * sizeof(int));
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 2 * 5 * sizeof(int));
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks<'x', 'y'>(), state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(7, 3, 2)) == 2 * 5 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_t<'x', 'y', 'z', vector_t<'x', scalar<int>>>, 'x',
									state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
}

TEST_CASE("into_blocks_static_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'y', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'z', state<>>);

	STATIC_REQUIRE(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'x',
		state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<0>)) == 3 * 16 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'y',
		state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>>);
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<0>)) == 2 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'z',
		state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>>);
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<0>)) == 0);

	STATIC_REQUIRE(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'x',
		state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<1>>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<1>)) == 3 * 16 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'y',
		state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<1>>>>);
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<1>)) == 2 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<into_blocks_static_t<'x', 'z', 'x', 'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 16>>, 'z',
		state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<1>>>>);
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_static<'x', 'z', 'x', 'y'>(lit<16>), state<state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>, state_item<index_in<'z'>, lit_t<0>>>(3, 2, lit<1>)) == 0);
}

TEST_CASE("into_blocks_dynamic_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', scalar<int>>, 'z', state<>>);

	STATIC_REQUIRE(HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x',
		state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_dynamic<'x', 'x', 'y', 'z'>(),
		state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 3 * 7 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y',
		state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_dynamic<'x', 'x', 'y', 'z'>(),
		state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 2 * sizeof(int));
	STATIC_REQUIRE(HasOffsetAlong<into_blocks_dynamic_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z',
		state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(35) ^ into_blocks_dynamic<'x', 'x', 'y', 'z'>(),
		state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>(5, 3, 2)) == 0);
}

TEST_CASE("merge_blocks_t", "[offset_along]") {
	STATIC_REQUIRE(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z', state<>>);

	STATIC_REQUIRE(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'x',
		state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'y',
		state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'z',
		state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'x', 'y', 'z'>(),
		state<state_item<index_in<'z'>, std::size_t>>(2)) == 2 * sizeof(int));
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'x', 'y', 'z'>(),
		state<state_item<index_in<'z'>, std::size_t>>(3)) == 3 * sizeof(int));
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'x', 'y', 'z'>(),
		state<state_item<index_in<'z'>, std::size_t>>(5)) == 5 * sizeof(int));
	STATIC_REQUIRE(!HasOffsetAlong<merge_blocks_t<'y', 'x', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'x',
		state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!HasOffsetAlong<merge_blocks_t<'y', 'x', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'y',
		state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(HasOffsetAlong<merge_blocks_t<'y', 'x', 'z', set_length_t<'x', set_length_t<'y', vector_t<'x', vector_t<'y', scalar<int>>>, std::size_t>, std::size_t>>, 'z',
		state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'y', 'x', 'z'>(),
		state<state_item<index_in<'z'>, std::size_t>>(2)) == 2 * 5 * sizeof(int));
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'y', 'x', 'z'>(),
		state<state_item<index_in<'z'>, std::size_t>>(3)) == 1 * sizeof(int));
	STATIC_REQUIRE(offset_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(3) ^ set_length<'y'>(5) ^ merge_blocks<'y', 'x', 'z'>(),
		state<state_item<index_in<'z'>, std::size_t>>(5)) == (2 * 5 + 1) * sizeof(int));
}

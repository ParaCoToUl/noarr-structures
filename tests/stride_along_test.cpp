#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/introspection/stride_along.hpp>

using namespace noarr;

TEST_CASE("bcast_t", "[stride_along]") {
	STATIC_REQUIRE(HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ bcast<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == 0);
	STATIC_REQUIRE(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<bcast_t<'x', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("vector_t", "[stride_along]") {
	STATIC_REQUIRE(HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) ==
				sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<vector_t<'x', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(
		HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
									state<state_item<length_in<'y'>, std::size_t>>(42)) == sizeof(int));
	STATIC_REQUIRE(
		!HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
									state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(
										42, 42)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
									state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(
									5, 42)) == 42 * sizeof(int));
}
TEST_CASE("scalar", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<scalar<int>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<scalar<int>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}
TEST_CASE("fix_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(
		!HasStrideAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<fix_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<fix_t<'x', vector_t<'y', scalar<int>>, std::size_t>, 'y',
								state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'y'>(scalar<int>() ^ vector<'y'>() ^ fix<'x'>(0),
									state<state_item<length_in<'y'>, std::size_t>>(0)) == sizeof(int));
}

TEST_CASE("set_length_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(HasStrideAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ set_length<'x'>(0), state<>()) == 0);
	STATIC_REQUIRE(HasStrideAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(1), state<>()) == sizeof(int));
}

TEST_CASE("rename_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ rename<'x', 'y'>(),
									state<state_item<length_in<'y'>, std::size_t>>(42)) == sizeof(int));
}

TEST_CASE("join_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y',
								state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ vector<'y'>() ^ join<'x', 'y', 'z'>(),
									state<state_item<length_in<'z'>, std::size_t>>(42)) == sizeof(int) + 42 * sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z',
								state<state_item<length_in<'z'>, std::size_t>, state_item<index_in<'z'>, std::size_t>>>);
}

TEST_CASE("shift_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<shift_t<'x', scalar<int>, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<shift_t<'x', scalar<int>, std::size_t>, 'x',
								state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ shift<'x'>(4), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<shift_t<'y', vector_t<'x', scalar<int>>, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ shift<'y'>(4), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'y',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<shift_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'x',
								state<>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ shift<'x'>(4), state<>()) == sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<shift_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'y',
								state<>>);
}

TEST_CASE("slice_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
								state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ slice<'x'>(4, 2), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<slice_t<'y', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ slice<'y'>(4, 3), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'y',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<slice_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x',
								state<>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ slice<'x'>(4, 4), state<>()) == sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<slice_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'y',
								state<>>);
}

TEST_CASE("span_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
								state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ span<'x'>(4, 6), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<span_t<'y', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ span<'y'>(4, 8), state<state_item<length_in<'x'>, std::size_t>>(42)) == sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'y',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<span_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x',
								state<>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ span<'x'>(4, 7), state<>()) == sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<span_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'y',
								state<>>);
}

TEST_CASE("step_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
								state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'x'>(4, 6), state<state_item<length_in<'x'>, std::size_t>>(42)) == 6 * sizeof(int));
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'x'>(4, 6) ^ step<'x'>(0, 2), state<state_item<length_in<'x'>, std::size_t>>(42)) == 2 * 6 * sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<step_t<'y', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ step<'y'>(4, 8), state<state_item<length_in<'x'>, std::size_t>>(42)) == 8 * sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'y',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<step_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x',
								state<>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ step<'x'>(4, 7), state<>()) == 7 * sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<step_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'y',
								state<>>);
}

TEST_CASE("reverse_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<reverse_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<reverse_t<'x', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<reverse_t<'x', scalar<int>>, 'x',
								state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == -static_cast<int>(sizeof(int)));
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'x'>() ^ reverse<'x'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == static_cast<int>(sizeof(int)));
	STATIC_REQUIRE(HasStrideAlong<reverse_t<'y', vector_t<'x', scalar<int>>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ reverse<'y'>(), state<state_item<length_in<'x'>, std::size_t>>(42)) == static_cast<int>(sizeof(int)));
	STATIC_REQUIRE(!HasStrideAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'y',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<reverse_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x',
								state<>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'x'>(), state<>()) == -static_cast<int>(sizeof(int)));
	STATIC_REQUIRE(!HasStrideAlong<reverse_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y',
								state<>>);
}

TEST_CASE("into_blocks_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(
		!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);

	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
								state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'x', 'y'>(),
									state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>(
										42, 5)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'x'>(scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'x', 'y'>(),
									state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>(
										42, 5)) == 42 * sizeof(int));
}

TEST_CASE("into_blocks_static_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'x', state<state_item<index_in<'y'>, lit_t<0>>>>);
	STATIC_REQUIRE(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'z', state<state_item<index_in<'y'>, lit_t<0>>>>);
	STATIC_REQUIRE(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
	state<state_item<index_in<'y'>, lit_t<0>>>()) == 4 * sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'w', state<state_item<index_in<'y'>, lit_t<0>>>>);
	STATIC_REQUIRE(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
	state<state_item<index_in<'y'>, lit_t<0>>>()) == sizeof(int));
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'y', state<>>);

	STATIC_REQUIRE(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'z', state<state_item<index_in<'y'>, lit_t<1>>>>);
	STATIC_REQUIRE(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
	state<state_item<index_in<'y'>, lit_t<1>>>()) == 2 * sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'w', state<state_item<index_in<'y'>, lit_t<1>>>>);
	STATIC_REQUIRE(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>),
	state<state_item<index_in<'y'>, lit_t<1>>>()) == sizeof(int));
}

TEST_CASE("into_blocks_dynamic_t", "[stride_along]") {
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<>>);
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<>>);
	STATIC_REQUIRE(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<>()) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<>>);
	STATIC_REQUIRE(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<>()) == 0);

	STATIC_REQUIRE(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'x'>, std::size_t>>(7)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'x'>, std::size_t>>(7)) == 0);

	STATIC_REQUIRE(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'y'>, std::size_t>>(7)) == 6 * sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'y'>, std::size_t>>(7)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'y'>, std::size_t>>(7)) == 0);

	STATIC_REQUIRE(!HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(7)) == 7 * sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(7)) == sizeof(int));
	STATIC_REQUIRE(HasStrideAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(stride_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(7)) == 0);
}

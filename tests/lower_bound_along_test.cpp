#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/introspection/lower_bound_along.hpp>

using namespace noarr;

TEST_CASE("bcast_t", "[lower_bound_along]") {
	STATIC_REQUIRE(HasLowerBoundAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ bcast<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>>(42)) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ bcast<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>>(42), 1, 2) == 0 * sizeof(int));
	STATIC_REQUIRE(!HasLowerBoundAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<bcast_t<'x', scalar<int>>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("vector_t", "[lower_bound_along]") {
	STATIC_REQUIRE(HasLowerBoundAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>>(42)) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>(),
										state<state_item<length_in<'x'>, std::size_t>>(42), 2, 3) == 2 * sizeof(int));
	STATIC_REQUIRE(!HasLowerBoundAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<vector_t<'x', scalar<int>>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(
		HasLowerBoundAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
										state<state_item<length_in<'y'>, std::size_t>>(42)) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
										state<state_item<length_in<'y'>, std::size_t>>(42), 2, 3) == 2 * sizeof(int));
	STATIC_REQUIRE(!HasLowerBoundAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
									state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		HasLowerBoundAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
						state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(
					scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
					state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(42, 42)) ==
				0 * sizeof(int));
	STATIC_REQUIRE(
		HasLowerBoundAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
						state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(
					scalar<int>() ^ vector<'y'>() ^ vector<'x'>(),
					state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>(5, 42)) == 0 * sizeof(int));
}

TEST_CASE("tuple_t", "[lower_bound_along]") {
	STATIC_REQUIRE(HasLowerBoundAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(pack(scalar<int>(), scalar<int>()) ^ tuple<'x'>(), state<>()) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(pack(scalar<int>(), scalar<int>()) ^ tuple<'x'>() ^ shift<'x'>(lit<1>), state<>()) == 1 * sizeof(int));
	STATIC_REQUIRE(!HasLowerBoundAlong<tuple_t<'x', scalar<int>, scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("scalar", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<scalar<int>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<scalar<int>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("fix_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<fix_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
									state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<fix_t<'x', vector_t<'y', scalar<int>>, std::size_t>, 'y',
									state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'y'>() ^ fix<'x'>(0),
										state<state_item<length_in<'y'>, std::size_t>>(5)) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'y'>() ^ fix<'x'>(0),
										state<state_item<length_in<'y'>, std::size_t>>(5), 3, 4) == 3 * sizeof(int));
}

TEST_CASE("set_length_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(HasLowerBoundAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ set_length<'x'>(5), state<>()) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ bcast<'x'>() ^ set_length<'x'>(5), state<>(), 3, 4) == 0 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(5), state<>()) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(5), state<>(), 3, 4) == 3 * sizeof(int));
}

TEST_CASE("rename_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'x'>() ^ rename<'x', 'y'>(), state<state_item<length_in<'y'>, std::size_t>>(5)) == 0 * sizeof(int));
}

TEST_CASE("shift_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<shift_t<'x', scalar<int>, std::size_t>, 'x',
									state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<shift_t<'x', scalar<int>, std::size_t>, 'x',
									state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<shift_t<'x', scalar<int>, std::size_t>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<shift_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ shift<'x'>(5), state<>()) == 5 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ shift<'x'>(2) ^ shift<'x'>(3), state<>()) == 5 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ shift<'x'>(2) ^ shift<'x'>(3), state<>(), 1, 2) == 6 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<shift_t<'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ shift<'y'>(5), state<>()) == 0 * sizeof(int));
}

TEST_CASE("slice_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
									state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
									state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<slice_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ slice<'x'>(5, 10), state<>()) == 5 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ slice<'x'>(5, 10), state<>(), 2, 3) == 7 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<slice_t<'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ slice<'y'>(5, 10), state<>()) == 0 * sizeof(int));
}

TEST_CASE("span_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
									state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
									state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<span_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ span<'x'>(5, 10), state<>()) == 5 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ span<'x'>(5, 10), state<>(), 3, 4) == 8 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<span_t<'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ span<'y'>(5, 10), state<>()) == 0 * sizeof(int));
}

TEST_CASE("reverse_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<reverse_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<reverse_t<'x', scalar<int>>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<reverse_t<'x', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'x'>(), state<>()) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'x'>(), state<>(), 2, 3) == 39 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'x'>(), state<>(), 2, 4) == 38 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<reverse_t<'y', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'y'>(), state<>()) == 0 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'y'>(), state<>(), 2, 3) == 2 * sizeof(int));
	STATIC_REQUIRE(lower_bound_along<'x'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ reverse<'y'>(), state<>(), 2, 4) == 2 * sizeof(int));
}

TEST_CASE("into_blocks_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
									state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
									state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
							state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
							state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);

	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
									state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
									state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
						state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(
					scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'x', 'y'>(),
					state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>(42, 5)) == 0 * sizeof(int));
	STATIC_REQUIRE(
		HasLowerBoundAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
						state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'x'>(
					scalar<int>() ^ vector<'x'>() ^ into_blocks<'x', 'x', 'y'>(),
					state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>(42, 5)) == 0 * sizeof(int));
}

TEST_CASE("into_blocks_static_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', scalar<int>, std::integral_constant<std::size_t, 4>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', scalar<int>, std::integral_constant<std::size_t, 4>>, 'y', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', scalar<int>, std::integral_constant<std::size_t, 4>>, 'z', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', scalar<int>, std::integral_constant<std::size_t, 4>>, 'w', state<>>);

	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'x', state<>>);
	STATIC_REQUIRE(HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'y', state<>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>), state<>()) == 0 * sizeof(int));
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'z', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'w', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'x', state<state_item<index_in<'y'>, lit_t<0>>>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'y', state<state_item<index_in<'y'>, lit_t<0>>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'z', state<state_item<index_in<'y'>, lit_t<0>>>>);
	STATIC_REQUIRE(lower_bound_along<'z'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>), state<state_item<index_in<'y'>, lit_t<0>>>(lit<0>)) == 0 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<into_blocks_static_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::integral_constant<std::size_t, 4>>, 'w', state<state_item<index_in<'y'>, lit_t<0>>>>);
	STATIC_REQUIRE(lower_bound_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_static<'x', 'y', 'z', 'w'>(lit<4>), state<state_item<index_in<'y'>, lit_t<0>>>(lit<0>)) == 0 * sizeof(int));
}

TEST_CASE("into_blocks_dynamic_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'z', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'w', state<>>);

	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'z', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<>>);

	STATIC_REQUIRE(!HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'y'>, std::size_t>>(5)) == 0 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'y', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'y'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(5)) == 0 * sizeof(int));
	STATIC_REQUIRE(HasLowerBoundAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>>, 'w', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(lower_bound_along<'w'>(scalar<int>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ into_blocks_dynamic<'x', 'y', 'z', 'w'>(), state<state_item<length_in<'z'>, std::size_t>>(5)) == 0 * sizeof(int));
}

TEST_CASE("merge_blocks_t", "[lower_bound_along]") {
	STATIC_REQUIRE(!HasLowerBoundAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z', state<>>);

	STATIC_REQUIRE(!HasLowerBoundAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'y', vector_t<'x', scalar<int>>>, std::size_t>, std::size_t>>, 'x', state<>>);
	STATIC_REQUIRE(!HasLowerBoundAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'y', vector_t<'x', scalar<int>>>, std::size_t>, std::size_t>>, 'y', state<>>);
	STATIC_REQUIRE(HasLowerBoundAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'y', vector_t<'x', scalar<int>>>, std::size_t>, std::size_t>>, 'z', state<>>);
	STATIC_REQUIRE(lower_bound_along<'z'>(scalar<int>() ^ vector<'y'>() ^ vector<'x'>() ^ set_length<'x'>(42) ^ set_length<'y'>(5) ^ merge_blocks<'x', 'y', 'z'>(), state<>()) == 0 * sizeof(int));
}

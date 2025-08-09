#include <noarr_test/macros.hpp>

#include <cstddef>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/introspection/uniform_along.hpp>

using namespace noarr;

TEST_CASE("bcast_t", "[uniform_along]") {
	STATIC_REQUIRE(IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<bcast_t<'x', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("vector_t", "[uniform_along]") {
	STATIC_REQUIRE(IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<vector_t<'x', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(
		IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'y',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<vector_t<'x', vector_t<'y', scalar<int>>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
}

TEST_CASE("scalar", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<scalar<int>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<scalar<int>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<scalar<int>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<scalar<int>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("fix_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(
		!IsUniformAlong<fix_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<fix_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<fix_t<'x', vector_t<'y', scalar<int>>, std::size_t>, 'y',
								state<state_item<length_in<'y'>, std::size_t>>>);
}

TEST_CASE("set_length_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<set_length_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(IsUniformAlong<set_length_t<'x', bcast_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(IsUniformAlong<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
}


TEST_CASE("rename_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'y', state<>>);
	STATIC_REQUIRE(IsUniformAlong<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
}

TEST_CASE("join_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'y', state<state_item<length_in<'y'>, std::size_t>, state_item<index_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<index_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<join_t<vector_t<'x', vector_t<'y', scalar<int>>>, 'x', 'y', 'z'>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
}

TEST_CASE("shift_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<shift_t<'x', scalar<int>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!IsUniformAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(IsUniformAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("slice_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(IsUniformAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("span_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(IsUniformAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("step_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', scalar<int>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(IsUniformAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("reverse_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', scalar<int>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);

	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<>>);
	STATIC_REQUIRE(IsUniformAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<reverse_t<'x', vector_t<'x', scalar<int>>>, 'x', state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>);
}

TEST_CASE("into_blocks_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(
		!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(
		!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x', state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', scalar<int>>, 'y',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);

	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
								state<state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
								state<state_item<length_in<'y'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'y',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<into_blocks_t<'x', 'x', 'y', vector_t<'x', scalar<int>>>, 'x',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'x'>, std::size_t>>>);
}

TEST_CASE("into_blocks_static_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'y', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', scalar<int>, std::size_t>, 'z', state<>>);

	STATIC_REQUIRE(!IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', vector_t<'x', scalar<int>>, std::size_t>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', vector_t<'x', scalar<int>>, std::size_t>, 'y', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', vector_t<'x', scalar<int>>, std::size_t>, 'z', state<>>);
	STATIC_REQUIRE(IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'x',
								state<>>);
	STATIC_REQUIRE(IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'y',
								state<>>);
	STATIC_REQUIRE(IsUniformAlong<into_blocks_static_t<'x', 'x', 'y', 'z', set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, std::size_t>, 'z',
								state<>>);
}

TEST_CASE("into_blocks_dynamic_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'z', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'w', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'y',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'z',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', scalar<int>>, 'w',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);

	STATIC_REQUIRE(IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', vector_t<'x', scalar<int>>>, 'y',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', vector_t<'x', scalar<int>>>, 'z',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(IsUniformAlong<into_blocks_dynamic_t<'x', 'y', 'z', 'w', vector_t<'x', scalar<int>>>, 'w',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);
}

TEST_CASE("merge_blocks_t", "[uniform_along]") {
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'y', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z', state<state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z',
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>);
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', scalar<int>>, 'z',
								state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>);

	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'y', vector_t<'x', scalar<int>>>, std::size_t>, std::size_t>>, 'x', state<>>);
	STATIC_REQUIRE(!IsUniformAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'y', vector_t<'x', scalar<int>>>, std::size_t>, std::size_t>>, 'y', state<>>);
	STATIC_REQUIRE(IsUniformAlong<merge_blocks_t<'x', 'y', 'z', set_length_t<'x', set_length_t<'y', vector_t<'y', vector_t<'x', scalar<int>>>, std::size_t>, std::size_t>>, 'z', state<>>);
}

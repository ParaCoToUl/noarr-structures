#include <noarr_test/macros.hpp>

#include <cstddef>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/introspection/contiguous.hpp>

namespace noarr::helpers {

TEST_CASE("bcast_t", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<scalar<int>, state<>>::value);
}

TEST_CASE("vector", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<vector_t<'x', scalar<int>>, state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<vector_t<'x', scalar<int>>, state<>>::value);
}

TEST_CASE("tuple", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>>,
								state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>>, state<>>::value);

	STATIC_REQUIRE(is_contiguous<tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>>,
								state<state_item<index_in<'t'>, lit_t<0>>>>::value);
	STATIC_REQUIRE(!is_contiguous<tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>>,
								state<state_item<index_in<'t'>, lit_t<1>>>>::value);
}

TEST_CASE("bcast", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<bcast_t<'x', scalar<int>>, state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<bcast_t<'x', scalar<int>>, state<>>::value);
}

TEST_CASE("fix", "[contiguous]") {
	STATIC_REQUIRE(
		is_contiguous<fix_t<'t', tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>>, lit_t<0>>, state<>>::value);
	STATIC_REQUIRE(
		!is_contiguous<fix_t<'t', tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>>, lit_t<1>>, state<>>::value);
}

TEST_CASE("set_length", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<set_length_t<'x', vector_t<'x', scalar<int>>, std::size_t>, state<>>::value);
	STATIC_REQUIRE(is_contiguous<set_length_t<'x', tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>>, std::size_t>,
								state<>>::value);
}

TEST_CASE("reorder", "[contiguous]") {
	STATIC_REQUIRE(
		is_contiguous<reorder_t<vector_t<'y', vector_t<'x', scalar<int>>>, 'x'>,
					state<state_item<length_in<'x'>, std::size_t>, state_item<length_in<'y'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<reorder_t<vector_t<'y', vector_t<'x', scalar<int>>>, 'x'>, state<>>::value);
}

TEST_CASE("hoist", "[contiguous]") {
	STATIC_REQUIRE(
		is_contiguous<hoist_t<'x', vector_t<'x', scalar<int>>>, state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<hoist_t<'x', vector_t<'x', scalar<int>>>, state<>>::value);
}

TEST_CASE("rename", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>,
								state<state_item<length_in<'y'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<rename_t<vector_t<'x', scalar<int>>, 'x', 'y'>, state<>>::value);
}

TEST_CASE("join", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<join_t<vector_t<'y', vector_t<'x', scalar<int>>>, 'x', 'y', 'z'>,
								state<state_item<length_in<'z'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<join_t<vector_t<'y', vector_t<'x', scalar<int>>>, 'x', 'y', 'z'>, state<>>::value);
}

TEST_CASE("shift", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>,
								state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<shift_t<'x', vector_t<'x', scalar<int>>, std::size_t>, state<>>::value);
}

TEST_CASE("slice", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>,
								state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<slice_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, state<>>::value);
}

TEST_CASE("span", "[contiguous]") {
	STATIC_REQUIRE(is_contiguous<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>,
								state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<span_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, state<>>::value);
}

TEST_CASE("step", "[contiguous]") {
	STATIC_REQUIRE(
		is_contiguous<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>,
					state<state_item<length_in<'x'>, std::size_t>, state_item<index_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>,
								state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<step_t<'x', vector_t<'x', scalar<int>>, std::size_t, std::size_t>, state<>>::value);
}

TEST_CASE("reverse", "[contiguous]") {
	STATIC_REQUIRE(
		is_contiguous<reverse_t<'x', vector_t<'x', scalar<int>>>, state<state_item<length_in<'x'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<reverse_t<'x', vector_t<'x', scalar<int>>>, state<>>::value);
}

TEST_CASE("into_blocks", "[contiguous]") {
	STATIC_REQUIRE(
		is_contiguous<into_blocks_t<'x', 'y', 'z', vector_t<'x', scalar<int>>>,
					state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>::value);
	STATIC_REQUIRE(is_contiguous<into_blocks_t<'x', 'y', 'z', vector_t<'x', scalar<int>>>,
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>,
									state_item<index_in<'y'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<into_blocks_t<'x', 'y', 'z', vector_t<'x', scalar<int>>>,
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>,
									state_item<index_in<'z'>, std::size_t>>>::value);
}

TEST_CASE("into_blocks_dynamic", "[contiguous]") {
	STATIC_REQUIRE(
		is_contiguous<into_blocks_dynamic_t<'x', 'y', 'z', 'w', vector_t<'x', scalar<int>>>,
					state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>>>::value);
	STATIC_REQUIRE(is_contiguous<into_blocks_dynamic_t<'x', 'y', 'z', 'w', vector_t<'x', scalar<int>>>,
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>,
									state_item<index_in<'y'>, std::size_t>>>::value);
	STATIC_REQUIRE(!is_contiguous<into_blocks_dynamic_t<'x', 'y', 'z', 'w', vector_t<'x', scalar<int>>>,
								state<state_item<length_in<'y'>, std::size_t>, state_item<length_in<'z'>, std::size_t>,
									state_item<index_in<'z'>, std::size_t>>>::value);

}

} // namespace noarr::helpers

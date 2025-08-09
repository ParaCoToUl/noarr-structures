#include <noarr_test/macros.hpp>

#include <cstddef>

#include <type_traits>

#include <noarr/structures_extended.hpp>

using namespace noarr;

TEST_CASE("reassemble: array ^ array", "[reassemble]") {
	array_t<'x', 10, array_t<'y', 20, scalar<int>>> array_x_array;

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y', 'x'>, array_t<'y', 20, array_t<'x', 10, scalar<int>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y', 'x'>())::signature,     array_t<'y', 20, array_t<'x', 10, scalar<int>>>::signature>);
	STATIC_REQUIRE(               decltype(array_x_array ^ reorder<'y', 'x'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y'>, array_t<'y', 20, scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y'>())::signature,     array_t<'y', 20, scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'y'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("reassemble: array ^ vector", "[reassemble]") {
	array_t<'x', 10, vector_t<'y', scalar<int>>> array_x_array;

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y', 'x'>, vector_t<'y', array_t<'x', 10, scalar<int>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y', 'x'>())::signature,     vector_t<'y', array_t<'x', 10, scalar<int>>>::signature>);
	STATIC_REQUIRE(               decltype(array_x_array ^ reorder<'y', 'x'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y'>, vector_t<'y', scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y'>())::signature,     vector_t<'y', scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'y'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("trivial reassemble: array ^ array", "[reassemble]") {
	array_t<'x', 10, array_t<'y', 20, scalar<int>>> array_x_array;

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x', 'y'>, array_t<'x', 10, array_t<'y', 20, scalar<int>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x', 'y'>())::signature,     array_t<'x', 10, array_t<'y', 20, scalar<int>>>::signature>);
	STATIC_REQUIRE(               decltype(array_x_array ^ reorder<'x', 'y'>())::complete       );
}

TEST_CASE("tuple reassemble: (int * float) ^ array", "[reassemble]") {
	array_t<'x', 10, tuple_t<'t', scalar<int>, scalar<float>>> array_x_array;

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x', 't'>, array_t<'x', 10, tuple_t<'t', scalar<int>, scalar<float>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x', 't'>())::signature,     array_t<'x', 10, tuple_t<'t', scalar<int>, scalar<float>>>::signature>);
	STATIC_REQUIRE(               decltype(array_x_array ^ reorder<'x', 't'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't', 'x'>, tuple_t<'t', array_t<'x', 10, scalar<int>>, array_t<'x', 10, scalar<float>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t', 'x'>())::signature,     tuple_t<'t', array_t<'x', 10, scalar<int>>, array_t<'x', 10, scalar<float>>>::signature>);
	STATIC_REQUIRE(               decltype(array_x_array ^ reorder<'t', 'x'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't'>, tuple_t<'t', scalar<void>, scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t'>())::signature,     tuple_t<'t', scalar<void>, scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'t'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("tuple reassemble: (int ^ array) * (float ^ array)", "[reassemble]") {
	tuple_t<'t', array_t<'x', 10, scalar<int>>, array_t<'x', 10, scalar<float>>> array_x_array;

	/* should not compile: "Tuple indices must not be omitted or moved down"
	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x', 't'>, array_t<'x', 10, tuple_t<'t', scalar<int>, scalar<float>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x', 't'>())::signature,     array_t<'x', 10, tuple_t<'t', scalar<int>, scalar<float>>>::signature>);
	STATIC_REQUIRE(               decltype(array_x_array ^ reorder<'x', 't'>())::complete       );*/

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't', 'x'>, tuple_t<'t', array_t<'x', 10, scalar<int>>, array_t<'x', 10, scalar<float>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t', 'x'>())::signature,     tuple_t<'t', array_t<'x', 10, scalar<int>>, array_t<'x', 10, scalar<float>>>::signature>);
	STATIC_REQUIRE(               decltype(array_x_array ^ reorder<'t', 'x'>())::complete       );

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't'>, tuple_t<'t', scalar<void>, scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t'>())::signature,     tuple_t<'t', scalar<void>, scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'t'>())::complete       );

	/* should not compile: "Tuple indices must not be omitted or moved down"
	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array_t<'x', 10, scalar<void>>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );*/

	STATIC_REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
	STATIC_REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("hoist: array ^ array", "[reassemble]") {
	array_t<'x', 10, array_t<'y', 20, scalar<int>>> array_x_array;

	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ hoist<'x'>())::signature, array_t<'x', 10, array_t<'y', 20, scalar<int>>>::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ hoist<'y'>())::signature, array_t<'y', 20, array_t<'x', 10, scalar<int>>>::signature>);
}

TEST_CASE("hoist hoist == double hoist", "[reassemble]") {
	array_t<'x', 10, array_t<'y', 20, scalar<int>>> array_x_array;

	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ hoist<'x'>() ^ hoist<'y'>())::signature, decltype(array_x_array ^ hoist<'y', 'x'>())::signature>);
	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ hoist<'y'>() ^ hoist<'x'>())::signature, decltype(array_x_array ^ hoist<'x', 'y'>())::signature>);
}

template<char Dim, class T>
using dynarray = set_length_t<Dim, vector_t<Dim, T>, std::size_t>;

TEST_CASE("strip mine", "[shortcuts blocks reassemble]") {
	array_t<'x', 10, array_t<'y', 20, scalar<int>>> array_x_array;

	STATIC_REQUIRE(std::is_same_v<decltype(array_x_array ^ strip_mine<'y', 'a', 'b'>(5))::signature, dynarray<'a', array_t<'x', 10, dynarray<'b', scalar<int>>>>::signature>);
}

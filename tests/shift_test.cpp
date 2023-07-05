#include <catch2/catch_test_macros.hpp>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/structs/slice.hpp>

TEST_CASE("Fixes and shifts", "[offsets]") {
	auto v = noarr::vector_t<'x', noarr::scalar<float>>();
	auto v2 = noarr::array_t<'y', 20'000, noarr::vector_t<'x', noarr::scalar<float>>>();

	auto v_sized = v ^ noarr::set_length<'x'>(20);
	auto v2_sized = v2 ^ noarr::set_length<'x'>(10'000);

	SECTION("check offsets") {
		// trivial case
		REQUIRE((v_sized | noarr::offset<'x'>(10)) == (v_sized ^ noarr::shift<'x'>(10) ^ noarr::fix<'x'>(0) | noarr::offset()));
		
		// composite case for one dimension
		REQUIRE((v_sized | noarr::offset<'x'>(10)) == (v_sized ^ noarr::shift<'x'>(5) ^ noarr::fix<'x'>(5) | noarr::offset()));

		// trivial case for composite structure
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'x'>(10) ^ noarr::fix<'x'>(0) ^ noarr::shift<'y'>(20) ^ noarr::fix<'y'>(0) | noarr::offset()));
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'y'>(20) ^ noarr::fix<'y'>(0) ^ noarr::shift<'x'>(10) ^ noarr::fix<'x'>(0) | noarr::offset()));
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'y'>(20) ^ noarr::shift<'x'>(10) ^ noarr::fix<'y'>(0) ^ noarr::fix<'x'>(0) | noarr::offset()));

		// composite case for composite structure
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'x'>(5) ^ noarr::fix<'x'>(5) ^ noarr::shift<'y'>(15) ^ noarr::fix<'y'>(5) | noarr::offset()));
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'y'>(15) ^ noarr::fix<'y'>(5) ^ noarr::shift<'x'>(5) ^ noarr::fix<'x'>(5) | noarr::offset()));
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'y'>(15) ^ noarr::shift<'x'>(5) ^ noarr::fix<'y'>(5) ^ noarr::fix<'x'>(5) | noarr::offset()));

		// simplified syntax
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'y', 'x'>(20, 10) ^ noarr::fix<'x', 'y'>(0, 0) | noarr::offset()));
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'x', 'y'>(5, 15) ^ noarr::fix<'x', 'y'>(5, 5) | noarr::offset()));
		REQUIRE((v2_sized | noarr::offset<'x', 'y'>(10, 20)) == (v2_sized ^ noarr::shift<'y', 'x'>(15, 5) ^ noarr::fix<'x', 'y'>(5, 5) | noarr::offset()));
	}
}

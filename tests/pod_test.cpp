#include <catch2/catch_test_macros.hpp>

#include <noarr/structures_extended.hpp>
#include "noarr_test_defs.hpp"

using namespace noarr;

TEST_CASE("Simplicity", "[low-lvl]") {
	SECTION("array is simple")
		REQUIRE(noarr_test::is_simple<array_t<'x', 100, scalar<int>>>);

	SECTION("vector is simple")
		REQUIRE(noarr_test::is_simple<vector_t<'x', scalar<int>>>);

	SECTION("tuple is simple")
		REQUIRE(noarr_test::is_simple<tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>, array_t<'y', 100, scalar<int>>>>);
}

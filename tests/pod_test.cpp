#include <catch2/catch_test_macros.hpp>

#include <noarr/structures.hpp>
#include "noarr_test_defs.hpp"

using namespace noarr;

TEST_CASE("Simplicity", "[low-lvl]") {
	SECTION("array is simple")
		REQUIRE(noarr_test::is_simple<array<'x', 100, scalar<int>>>);

	SECTION("vector is simple")
		REQUIRE(noarr_test::is_simple<vector<'x', scalar<int>>>);

	SECTION("tuple is simple")
		REQUIRE(noarr_test::is_simple<tuple<'t', scalar<int>, vector<'x', scalar<int>>, array<'y', 100, scalar<int>>>>);
}

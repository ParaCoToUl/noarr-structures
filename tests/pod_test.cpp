#include <noarr_test/macros.hpp>

#include <noarr/structures_extended.hpp>
#include "noarr_test_defs.hpp"

using namespace noarr;

TEST_CASE("Simplicity", "[low-lvl]") {
	REQUIRE(noarr_test::is_simple<array_t<'x', 100, scalar<int>>>);
	REQUIRE(noarr_test::is_simple<vector_t<'x', scalar<int>>>);
	REQUIRE(noarr_test::is_simple<tuple_t<'t', scalar<int>, vector_t<'x', scalar<int>>, array_t<'y', 100, scalar<int>>>>);
}

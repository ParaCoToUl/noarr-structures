#include <catch2/catch.hpp>

#include "noarr/structures.hpp"
#include "noarr_test_defs.hpp"

using namespace noarr;

TEST_CASE("Simplicity", "[low-lvl]") {
    SECTION("array is simple")
        REQUIRE(noarr_test::is_simple<array<'x', 100, scalar<int>>>);

    SECTION("vector is simple")
        REQUIRE(noarr_test::is_simple<vector<'x', scalar<int>>>);

    SECTION("sized_vector is simple")
        REQUIRE(noarr_test::is_simple<sized_vector<'x', scalar<int>>>);

    SECTION("tuple is simple")
        REQUIRE(noarr_test::is_simple<tuple<'t', scalar<int>, vector<'x', scalar<int>>, array<'y', 100, scalar<int>>>>);

    SECTION("sfixed_dim sized_vector is simple")
        REQUIRE(noarr_test::is_simple<sfixed_dim<'x', sized_vector<'x', scalar<int>>, 10>>);

    SECTION("fixed_dim sized_vector is simple")
        REQUIRE(noarr_test::is_simple<fixed_dim<'x', sized_vector<'x', scalar<int>>>>);
}

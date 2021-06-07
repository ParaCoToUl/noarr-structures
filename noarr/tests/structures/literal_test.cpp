#include <catch2/catch.hpp>
//#include "noarr/structures.hpp"

#include <iostream>
#include <array>

#include "noarr/structures/structs.hpp"
#include "noarr/structures/funcs.hpp"
#include "noarr/structures/io.hpp"
#include "noarr/structures/struct_traits.hpp"
#include "noarr/structures/wrapper.hpp"

using namespace noarr::literals;
using namespace noarr;

TEST_CASE("Triviality", "[low-lvl]") {
    SECTION("array is trivial")
        REQUIRE(std::is_trivial<array<'x', 100, scalar<int>>>::value);

    SECTION("vector is trivial")
        REQUIRE(std::is_trivial<vector<'x', scalar<int>>>::value);

    SECTION("sized_vector is trivial")
        REQUIRE(std::is_trivial<sized_vector<'x', scalar<int>>>::value);

    SECTION("tuple is trivial")
        REQUIRE(std::is_trivial<tuple<'t', scalar<int>, vector<'x', scalar<int>>, array<'y', 100, scalar<int>>>>::value);

    SECTION("sfixed_dim sized_vector is trivial")
        REQUIRE(std::is_trivial<sfixed_dim<'x', sized_vector<'x', scalar<int>>, 10>>::value);

    SECTION("fixed_dim sized_vector is trivial")
        REQUIRE(std::is_trivial<fixed_dim<'x', sized_vector<'x', scalar<int>>>>::value);
}

TEST_CASE("Standard layout", "[low-lvl]") {
    SECTION("array is is standard layout")
        REQUIRE(std::is_standard_layout<array<'x', 100, scalar<int>>>::value);

    SECTION("vector is is standard layout")
        REQUIRE(std::is_standard_layout<vector<'x', scalar<int>>>::value);

    SECTION("sized_vector is is standard layout")
        REQUIRE(std::is_standard_layout<sized_vector<'x', scalar<int>>>::value);

    SECTION("tuple is is standard layout")
        REQUIRE(std::is_standard_layout<tuple<'t', scalar<int>, vector<'x', scalar<int>>, array<'y', 100, scalar<int>>>>::value);

    SECTION("sfixed_dim sized_vector is is standard layout")
        REQUIRE(std::is_standard_layout<sfixed_dim<'x', sized_vector<'x', scalar<int>>, 10>>::value);

    SECTION("fixed_dim sized_vector is is standard layout")
        REQUIRE(std::is_standard_layout<fixed_dim<'x', sized_vector<'x', scalar<int>>>>::value);
}

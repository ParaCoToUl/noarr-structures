#include <catch2/catch.hpp>

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

TEST_CASE("Literality test", "[low-lvl]") {
    constexpr auto test_array = array<'x', 100, scalar<int>>();
    test_array.size();
    constexpr auto test_vector = vector<'x', scalar<int>>();
    constexpr auto test_sized_vector = test_vector | set_length<'x'>(100);
    test_sized_vector.size();
    constexpr auto test_tuple = tuple<'t', scalar<int>, sized_vector<'x', scalar<int>>, array<'y', 100, scalar<int>>>();
    test_tuple.size();
    constexpr auto test_sfixed_dim = sfixed_dim<'x', sized_vector<'x', scalar<int>>, 10>();
    test_sfixed_dim.size();
    constexpr auto test_fixed_dim = fixed_dim<'x', sized_vector<'x', scalar<int>>>();
    test_fixed_dim.size();
}

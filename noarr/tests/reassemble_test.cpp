#include <catch2/catch.hpp>

#include "noarr/structures_extended.hpp"

using namespace noarr;

TEST_CASE("reassemble: array ^ array", "[reassemble]") {
    array<'x', 10, array<'y', 20, scalar<int>>> array_x_array;

    REQUIRE(std::is_same<decltype(array_x_array | reassemble<'x', 'y'>()), array<'y', 20, array<'x', 10, scalar<int>>>>::value);
}

TEST_CASE("reassemble: array ^ vector", "[reassemble]") {
    array<'x', 10, vector<'y', scalar<int>>> array_x_array;

    REQUIRE(std::is_same<decltype(array_x_array | reassemble<'x', 'y'>()), vector<'y', array<'x', 10, scalar<int>>>>::value);
}

TEST_CASE("trivial reassemble: vector * array (tuple)", "[reassemble]") {
    array<'x', 10, array<'y', 20, scalar<int>>> array_x_array;

    REQUIRE(std::is_same<decltype(array_x_array | reassemble<'x', 'x'>()), array<'x', 10, array<'y', 20, scalar<int>>>>::value);
}

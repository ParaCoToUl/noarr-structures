#include <catch2/catch.hpp>
#include "noarr/structures.hpp"

using namespace noarr;

TEST_CASE("Dummy method works", "[dummy]") {
    vector<'x', scalar<float>> v;
    auto vs = v | resize<'x'>(10); // transform
    REQUIRE(is_cube<decltype(vs)>::value == true);
    REQUIRE(vs.size() == 40);
}

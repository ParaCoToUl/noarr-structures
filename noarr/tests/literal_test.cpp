#include <catch2/catch.hpp>

#include "noarr/structures.hpp"

using namespace noarr;

TEST_CASE("Literality test", "[low-lvl]") {
    constexpr auto test_array = array<'x', 100, scalar<int>>();
    constexpr auto test_array_size = test_array.size();

    constexpr auto test_vector = vector<'x', scalar<int>>();
    constexpr auto test_sized_vector = test_vector | set_length<'x'>(100);
    constexpr auto test_sized_vector_size = test_sized_vector.size();

    constexpr auto test_tuple = tuple<'t', scalar<int>, sized_vector<'x', scalar<int>>, array<'y', 100, scalar<int>>>();
    constexpr auto test_tuple_size = test_tuple.size();

    constexpr auto test_sfixed_dim = sfixed_dim<'x', sized_vector<'x', scalar<int>>, 10>();
    constexpr auto test_sfixed_dim_size = test_sfixed_dim.size();

    constexpr auto test_fixed_dim = fixed_dim<'x', sized_vector<'x', scalar<int>>>();
    constexpr auto test_fixed_dim_size = test_fixed_dim.size();

    REQUIRE(std::integral_constant<std::size_t, (test_array_size + test_sized_vector_size + test_tuple_size + test_sfixed_dim_size + test_fixed_dim_size)>::value == 1204UL);
}

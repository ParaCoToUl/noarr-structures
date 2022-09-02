#include <catch2/catch.hpp>

#include "noarr/structures_extended.hpp"
#include "noarr/structures/reorder.hpp"
#include "noarr/structures/shortcuts.hpp"

using namespace noarr;

TEST_CASE("reassemble: array ^ array", "[reassemble]") {
    array<'x', 10, array<'y', 20, scalar<int>>> array_x_array;

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y', 'x'>, array<'y', 20, array<'x', 10, scalar<int>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y', 'x'>())::signature,     array<'y', 20, array<'x', 10, scalar<int>>>::signature>);
    REQUIRE(               decltype(array_x_array ^ reorder<'y', 'x'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array<'x', 10, scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array<'x', 10, scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y'>, array<'y', 20, scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y'>())::signature,     array<'y', 20, scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'y'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("reassemble: array ^ vector", "[reassemble]") {
    array<'x', 10, vector<'y', scalar<int>>> array_x_array;

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y', 'x'>, vector<'y', array<'x', 10, scalar<int>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y', 'x'>())::signature,     vector<'y', array<'x', 10, scalar<int>>>::signature>);
    REQUIRE(               decltype(array_x_array ^ reorder<'y', 'x'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array<'x', 10, scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array<'x', 10, scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'y'>, vector<'y', scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'y'>())::signature,     vector<'y', scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'y'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("trivial reassemble: array ^ array", "[reassemble]") {
    array<'x', 10, array<'y', 20, scalar<int>>> array_x_array;

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x', 'y'>, array<'x', 10, array<'y', 20, scalar<int>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x', 'y'>())::signature,     array<'x', 10, array<'y', 20, scalar<int>>>::signature>);
    REQUIRE(               decltype(array_x_array ^ reorder<'x', 'y'>())::complete       );
}

TEST_CASE("tuple reassemble: (int * float) ^ array", "[reassemble]") {
    array<'x', 10, tuple<'t', scalar<int>, scalar<float>>> array_x_array;

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x', 't'>, array<'x', 10, tuple<'t', scalar<int>, scalar<float>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x', 't'>())::signature,     array<'x', 10, tuple<'t', scalar<int>, scalar<float>>>::signature>);
    REQUIRE(               decltype(array_x_array ^ reorder<'x', 't'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't', 'x'>, tuple<'t', array<'x', 10, scalar<int>>, array<'x', 10, scalar<float>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t', 'x'>())::signature,     tuple<'t', array<'x', 10, scalar<int>>, array<'x', 10, scalar<float>>>::signature>);
    REQUIRE(               decltype(array_x_array ^ reorder<'t', 'x'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't'>, tuple<'t', scalar<void>, scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t'>())::signature,     tuple<'t', scalar<void>, scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'t'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array<'x', 10, scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array<'x', 10, scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("tuple reassemble: (int ^ array) * (float ^ array)", "[reassemble]") {
    tuple<'t', array<'x', 10, scalar<int>>, array<'x', 10, scalar<float>>> array_x_array;

    /* should not compile: "Tuple indices must not be omitted or moved down"
    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x', 't'>, array<'x', 10, tuple<'t', scalar<int>, scalar<float>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x', 't'>())::signature,     array<'x', 10, tuple<'t', scalar<int>, scalar<float>>>::signature>);
    REQUIRE(               decltype(array_x_array ^ reorder<'x', 't'>())::complete       );*/

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't', 'x'>, tuple<'t', array<'x', 10, scalar<int>>, array<'x', 10, scalar<float>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t', 'x'>())::signature,     tuple<'t', array<'x', 10, scalar<int>>, array<'x', 10, scalar<float>>>::signature>);
    REQUIRE(               decltype(array_x_array ^ reorder<'t', 'x'>())::complete       );

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 't'>, tuple<'t', scalar<void>, scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'t'>())::signature,     tuple<'t', scalar<void>, scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'t'>())::complete       );

    /* should not compile: "Tuple indices must not be omitted or moved down"
    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature, 'x'>, array<'x', 10, scalar<void>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<'x'>())::signature,     array<'x', 10, scalar<void>>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<'x'>())::complete       );*/

    REQUIRE(std::is_same_v<reassemble_sig<decltype(array_x_array)::signature>, scalar<void>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ reorder<>())::signature,   scalar<void>::signature>);
    REQUIRE(             ! decltype(array_x_array ^ reorder<>())::complete     );
}

TEST_CASE("hoist: array ^ array", "[reassemble]") {
    array<'x', 10, array<'y', 20, scalar<int>>> array_x_array;

    REQUIRE(std::is_same_v<decltype(array_x_array ^ hoist<'x'>())::signature, array<'x', 10, array<'y', 20, scalar<int>>>::signature>);
    REQUIRE(std::is_same_v<decltype(array_x_array ^ hoist<'y'>())::signature, array<'y', 20, array<'x', 10, scalar<int>>>::signature>);
}

using namespace noarr::literals;

template<char Dim, class T>
using dynarray = set_length_t<Dim, vector<Dim, T>, std::size_t>;

TEST_CASE("strip mine", "[shortcuts blocks reassemble]") {
    array<'x', 10, array<'y', 20, scalar<int>>> array_x_array;

    //decltype(array_x_array ^ strip_mine<'y', 'a', 'b'>(5))::signature i = 42;
    //vector<'a', array<'x', 10, vector<'b', scalar<int>>>>::signature j = 42;
    REQUIRE(std::is_same_v<decltype(array_x_array ^ strip_mine<'y', 'a', 'b'>(5))::signature, dynarray<'a', array<'x', 10, dynarray<'b', scalar<int>>>>::signature>);
    //REQUIRE(std::is_same_v<decltype(array_x_array ^ strip_mine<'y', 'a', 'b'>(5_idx))::signature, array<'a', 4, array<'x', 10, array<'b', 5, scalar<int>>>>::signature>);
}

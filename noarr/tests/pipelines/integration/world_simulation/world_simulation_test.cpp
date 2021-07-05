#include <catch2/catch.hpp>

#include <string>
#include <iostream>
#include <vector>

#include "world_simulation.hpp"

TEST_CASE("World simulation example", "[pipelines][integration][world_simulation]") {
    const std::size_t TARGET_ITERATIONS = 10;
    std::vector<std::int32_t> world_data = {1, 2, 3, 4, 5, 6};
    std::vector<std::int32_t> expected_world_data = {1024, 2048, 3072, 4096, 5120, 6144};
    
    SECTION("works via inheritance") {
        world_simulation_via_inheritance(world_data, TARGET_ITERATIONS);

        for (std::size_t i = 0; i < world_data.size(); ++i) {
            REQUIRE(expected_world_data[i] == world_data[i]);
        }
    }

    SECTION("works via builder") {
        world_simulation_via_builder(world_data, TARGET_ITERATIONS);

        for (std::size_t i = 0; i < world_data.size(); ++i) {
            REQUIRE(expected_world_data[i] == world_data[i]);
        }
    }
}

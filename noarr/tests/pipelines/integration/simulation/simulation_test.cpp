#include <catch2/catch.hpp>

#include <string>
#include <iostream>
#include <vector>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>

#include "SimulatorNode.hpp"

using namespace noarr::pipelines;

TEST_CASE("Simulation example", "[pipelines][integration][simulation]") {

    const std::size_t TARGET_ITERATIONS = 10;

    std::vector<std::int32_t> medium_data = {1, 2, 3, 4, 5, 6};
    std::vector<std::int32_t> expected_medium_data = {1024, 2048, 3072, 4096, 5120, 6144};

    auto medium_hub = Hub<std::size_t, std::int32_t>(
        sizeof(std::int32_t) * medium_data.size(),
        {
            {Device::HOST_INDEX, 2},
            {Device::DEVICE_INDEX, 1},
        }
    );
    auto simulator = SimulatorNode(TARGET_ITERATIONS, medium_data, medium_hub);

    // medium_hub.set_dataflow_strategy.to_link(simulator);

    auto scheduler = DebuggingScheduler();
    scheduler.add(medium_hub);
    scheduler.add(simulator);
    
    SECTION("runs to completion and returns proper result") {
        scheduler.run();

        for (std::size_t i = 0; i < medium_data.size(); ++i) {
            REQUIRE(expected_medium_data[i] == medium_data[i]);
        }
    }
}

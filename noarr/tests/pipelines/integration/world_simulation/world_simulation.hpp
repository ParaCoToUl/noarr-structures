#include <string>
#include <iostream>
#include <vector>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>

#include "WorldSimulatorNode.hpp"

using namespace noarr::pipelines;

void world_simulation(
    std::vector<std::int32_t>& world_data,
    std::size_t target_iterations
) {
    auto world_hub = Hub<std::size_t, std::int32_t>(
        sizeof(std::int32_t) * world_data.size(),
        {
            {Device::HOST_INDEX, 2},
            {Device::DEVICE_INDEX, 1},
        }
    );
    auto simulator_node = WorldSimulatorNode(
        target_iterations,
        world_data,
        world_hub
    );

    // world_hub.set_dataflow_strategy.to_link(simulator_node);

    auto scheduler = DebuggingScheduler();
    scheduler.add(world_hub);
    scheduler.add(simulator_node);

    scheduler.run();
}

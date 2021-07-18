#include <string>
#include <iostream>
#include <vector>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Envelope.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>
#include <noarr/pipelines/LambdaComputeNode.hpp>

#include "WorldSimulatorNode.hpp"

using namespace noarr::pipelines;

void world_simulation_via_inheritance(
    std::vector<std::int32_t>& world_data,
    std::size_t target_iterations
) {
    auto world_hub = Hub<std::size_t, std::int32_t>(sizeof(std::int32_t) * world_data.size());
    world_hub.allocate_envelopes(Device::HOST_INDEX, 1);

    auto simulator_node = WorldSimulatorNode(
        target_iterations,
        world_data,
        world_hub
    );

    auto scheduler = DebuggingScheduler();
    scheduler.add(world_hub);
    scheduler.add(simulator_node);

    scheduler.run();
}

void world_simulation_via_builder(
    std::vector<std::int32_t>& world_data,
    std::size_t target_iterations
) {
    std::size_t finished_iterations = 0;

    auto world_hub = Hub<std::size_t, std::int32_t>(sizeof(std::int32_t) * world_data.size());
    world_hub.allocate_envelopes(Device::HOST_INDEX, 1);

    auto simulator_node = LambdaComputeNode([&](auto& node){
        auto& world_link = node.link(world_hub.to_modify(Device::HOST_INDEX));

        node.initialize([&](){
            // load data from the variable into the hub
            Envelope<std::size_t, std::int32_t>& env = world_hub.push_new_chunk();

            env.structure = world_data.size();
            for (std::size_t i = 0; i < world_data.size(); ++i) {
                env.buffer[i] = world_data[i];
            }
        });

        node.can_advance([&](){
            return finished_iterations < target_iterations;
        });

        node.advance([&](){
            std::size_t n = world_link.envelope->structure;
            std::int32_t* items = world_link.envelope->buffer;

            for (std::size_t i = 0; i < n; ++i)
                items[i] *= 2;
            
            node.callback();
        });

        node.post_advance([&](){
            finished_iterations += 1;
        });

        node.terminate([&](){
            // pull the data from the hub back to the variable
            Envelope<std::size_t, std::int32_t>& env = world_hub.peek_latest_chunk();

            world_data.resize(env.structure);
            for (std::size_t i = 0; i < world_data.size(); ++i) {
                world_data[i] = env.buffer[i];
            }
        });
    });

    auto scheduler = DebuggingScheduler();
    scheduler.add(world_hub);
    scheduler.add(simulator_node);

    scheduler.run();
}

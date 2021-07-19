#include <cstddef>
#include <iostream>
#include <vector>

#include <noarr/pipelines/ComputeNode.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/Envelope.hpp>

using namespace noarr::pipelines;

class WorldSimulatorNode : public noarr::pipelines::ComputeNode {
private:
    std::size_t target_iterations;
    std::size_t finished_iterations;

    std::vector<std::int32_t>& world_data;
    Hub<std::size_t, std::int32_t>& world_hub;
    Link<std::size_t, std::int32_t>& world_link;

public:
    WorldSimulatorNode(
        std::size_t target_iterations,
        std::vector<std::int32_t>& world_data,
        Hub<std::size_t, std::int32_t>& world_hub
    ) :
        noarr::pipelines::ComputeNode(),
        target_iterations(target_iterations),
        finished_iterations(0),
        world_data(world_data),
        world_hub(world_hub),
        world_link(this->link(world_hub.to_modify(Device::HOST_INDEX)))
    { }

    void initialize() override {
        // load data from the variable into the hub

        Envelope<std::size_t, std::int32_t>& env = world_hub.push_new_chunk();

        env.structure = world_data.size();
        for (std::size_t i = 0; i < world_data.size(); ++i) {
            env.buffer[i] = world_data[i];
        }
    }

    bool can_advance() override {
        return finished_iterations < target_iterations;
    }

    void advance() override {
        std::size_t n = world_link.envelope->structure;
        std::int32_t* items = world_link.envelope->buffer;

        for (std::size_t i = 0; i < n; ++i)
            items[i] *= 2;

        this->callback();
    }

    void post_advance() override {
        finished_iterations += 1;
    }

    void terminate() override {
        // pull the data from the hub back to the variable

        Envelope<std::size_t, std::int32_t>& env = world_hub.peek_top_chunk();

        world_data.resize(env.structure);
        for (std::size_t i = 0; i < world_data.size(); ++i) {
            world_data[i] = env.buffer[i];
        }
    }
};

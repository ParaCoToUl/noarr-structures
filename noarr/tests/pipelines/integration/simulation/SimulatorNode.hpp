#include <cstddef>
#include <iostream>
#include <vector>

#include <noarr/pipelines/ComputeNode.hpp>
#include <noarr/pipelines/Hub.hpp>
#include <noarr/pipelines/Envelope.hpp>

using namespace noarr::pipelines;

class SimulatorNode : public noarr::pipelines::ComputeNode {
private:
    std::size_t target_iterations;
    std::size_t finished_iterations;

    std::vector<std::int32_t>& medium_data;
    Hub<std::size_t, std::int32_t>& medium_hub;
    Link<std::size_t, std::int32_t>& medium;

public:
    SimulatorNode(
        std::size_t _target_iterations,
        std::vector<std::int32_t>& _medium_data,
        Hub<std::size_t, std::int32_t>& _medium_hub
    ) :
        noarr::pipelines::ComputeNode(),
        target_iterations(_target_iterations),
        finished_iterations(0),
        medium_data(_medium_data),
        medium_hub(_medium_hub),
        medium(this->link(medium_hub.to_modify(Device::HOST_INDEX)))
    { }

    void initialize() override {
        // load data from the variable into the hub

        Envelope<std::size_t, std::int32_t>& env = medium_hub.push_new_chunk();

        env.structure = medium_data.size();
        for (std::size_t i = 0; i < medium_data.size(); ++i) {
            env.buffer[i] = medium_data[i];
        }
    }

    bool can_advance() override {
        return finished_iterations < target_iterations;
    }

    void advance() override {
        std::size_t n = medium.envelope->structure;
        std::int32_t* items = medium.envelope->buffer;

        for (std::size_t i = 0; i < n; ++i)
            items[i] *= 2;

        this->callback();
    }

    void post_advance() override {
        finished_iterations += 1;
    }

    void terminate() override {
        // pull the data from the hub back to the variable

        Envelope<std::size_t, std::int32_t>& env = medium_hub.peek_latest_chunk();

        medium_data.resize(env.structure);
        for (std::size_t i = 0; i < medium_data.size(); ++i) {
            medium_data[i] = env.buffer[i];
        }
    }
};

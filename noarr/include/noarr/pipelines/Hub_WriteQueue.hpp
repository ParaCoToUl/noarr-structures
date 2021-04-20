#ifndef NOARR_PIPELINES_HUB_WRITE_QUEUE_HPP
#define NOARR_PIPELINES_HUB_WRITE_QUEUE_HPP

#include <map>
#include <vector>

#include "Device.hpp"
#include "Port.hpp"
#include "Node.hpp"

namespace noarr {
namespace pipelines {
namespace hub {

/**
 * Implements the write queue of a hub
 */
template<typename Structure, typename BufferItem = void>
class WriteQueue : public Node {
public:
    /**
     * Ports (one for each producer) where freshly filled envelopes are received
     */
    std::vector<Port<Structure, BufferItem>> input_ports;

    /**
     * Ports (one for each device) where envelopes leave the queue
     */
    std::map<Device::index_t, Port<Structure, BufferItem>> output_ports;

    WriteQueue() : Node(typeid(WriteQueue).name()) { }

    //////////////
    // Node API //
    //////////////

    virtual void register_ports(std::function<void(UntypedPort*)> register_port) {
        for (auto& p : this->input_ports)
            register_port(&p);

        for (auto& kv : this->output_ports)
            register_port(&kv.second);
    };

    bool can_advance() override {
        return false; // TODO
    }

    void advance(std::function<void()> callback) override {
        callback(); // TODO
    }

    /////////////////////
    // Write queue API //
    /////////////////////

    /**
     * Creates a new port to which freshly filled envelopes can be sent
     */
    Port<Structure, BufferItem>& get_input_port(Device::index_t device_index) {
        this->input_ports.push_back(Port<Structure, BufferItem>(device_index));
        return this->input_ports.back();
    }
};

} // hub namespace
} // pipelines namespace
} // namespace noarr

#endif

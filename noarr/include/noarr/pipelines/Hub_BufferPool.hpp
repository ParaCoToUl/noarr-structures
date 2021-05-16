#ifndef NOARR_PIPELINES_HUB_BUFFER_POOL_HPP
#define NOARR_PIPELINES_HUB_BUFFER_POOL_HPP

#include <map>
#include <vector>

#include "Device.hpp"
#include "Port.hpp"
#include "Node.hpp"

namespace noarr {
namespace pipelines {
namespace hub {

/**
 * Holds a pool of empty buffers
 */
template<typename Structure, typename BufferItem = void>
class BufferPool : public Node {
public:
    /**
     * Ports (one for each device) where empty envelopes are returned from
     * the compute nodes
     */
    std::map<Device::index_t, Port<Structure, BufferItem>> input_ports;

    /**
     * Ports (one for each requester) where empty envelopes are sent
     * towards compute nodes to be filled
     */
    std::vector<Port<Structure, BufferItem>> output_ports;

    BufferPool() : Node(typeid(BufferPool).name()) { }

    //////////////
    // Node API //
    //////////////

    virtual void register_ports(std::function<void(UntypedPort*)> register_port) {
        for (auto& kv : this->input_ports)
            register_port(&kv.second);
        
        for (auto& p : this->output_ports)
            register_port(&p);
    };

    bool can_advance() override {
        return false; // TODO implement this
    }

    void advance(std::function<void()> callback) override {
        callback(); // TODO implement this
    }

    /////////////////////
    // Buffer pool API //
    /////////////////////

    /**
     * Prepares a port to which empty envelopes can be forwarded and they will
     * be added to the buffer pool
     */
    Port<Structure, BufferItem>& get_input_port(Device::index_t device_index) {
        if (!this->input_ports.count(device_index)) {
            this->input_ports[device_index] = Port<Structure, BufferItem>(device_index);
        }

        return this->input_ports[device_index];
    }

    /**
     * Creates a new port that will provide you with empty envelopes to use
     */
    Port<Structure, BufferItem>& create_output_port(Device::index_t device_index) {
        this->output_ports.push_back(Port<Structure, BufferItem>(device_index));
        return this->output_ports.back();
    }
};

} // hub namespace
} // pipelines namespace
} // namespace noarr

#endif

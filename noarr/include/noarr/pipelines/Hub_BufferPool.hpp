#ifndef NOARR_PIPELINES_HUB_BUFFER_POOL_HPP
#define NOARR_PIPELINES_HUB_BUFFER_POOL_HPP

#include <map>
#include <vector>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/Port.hpp"
#include "noarr/pipelines/Node.hpp"

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
    std::map<Device::index_t, Port<Structure, BufferItem>> return_ports;

    /**
     * Ports (one for each requester) where empty envelopes are sent
     * towards compute nodes to be filled
     */
    std::vector<Port<Structure, BufferItem>> send_ports;

    BufferPool() : Node(typeid(BufferPool).name()) { }

    virtual void register_ports(std::function<void(UntypedPort*)> register_port) {
        for (auto& kv : this->return_ports)
            register_port(&kv.second);
        
        for (auto& p : this->send_ports)
            register_port(&p);
    };

    bool can_advance() override {
        return false; // TODO
    }

    void advance(std::function<void()> callback) override {
        callback(); // TODO
    }
};

} // hub namespace
} // pipelines namespace
} // namespace noarr

#endif

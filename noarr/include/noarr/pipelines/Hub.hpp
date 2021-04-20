#ifndef NOARR_PIPELINES_HUB_HPP
#define NOARR_PIPELINES_HUB_HPP

#include <vector>

#include "UntypedPort.hpp"
#include "Device.hpp"
#include "CompositeNode.hpp"

#include "Hub_Link.hpp"
#include "Hub_BufferPool.hpp"
#include "Hub_WriteQueue.hpp"

namespace noarr {
namespace pipelines {

/**
 * Envelope hub is responsible for:
 * - envelope lifecycle
 * - envelope routing
 * - moving data between devices
 * It is the glue between compute nodes.
 */
template<typename Structure, typename BufferItem>
class Hub : public CompositeNode {
public:
    using Link = hub::Link<Structure, BufferItem>;

    /**
     * Holds empty buffers and distributes them to write links
     */
    hub::BufferPool<Structure, BufferItem> buffer_pool;

    /**
     * Holds freshly produced envelopes, as they arrive into the hub
     * TODO: merge with the DistributeQueue and rename to it
     */
    hub::WriteQueue<Structure, BufferItem> write_queue;

    // TODO: construction API to be figured out
    Hub() : buffer_pool() {
        this->register_constituent_node(this->buffer_pool);
    }

    ///////////////////////
    // Link creation API //
    ///////////////////////

private:
    std::vector<Link> links;

    Link& create_link(Device::index_t device_index, hub::LinkFlags flags) {
        auto link = Link(*this, flags, device_index, [&](
            Link& l, Port<Structure, BufferItem>& p
        ){
            this->attach_link_to_node_port(l, p);
        });

        this->links.push_back(std::move(link));

        return this->links.back();
    }

    void attach_link_to_node_port(Link& link, Port<Structure, BufferItem>& node_port) {
        hub::LinkFlags flags = link.flags();
        Device::index_t dev = link.device_index();
        
        if (flags == hub::LinkFlags::write) {
            auto& source_port = this->buffer_pool.get_output_port(dev);
            auto& target_port = this->write_queue.get_input_port(dev);
            source_port.send_processed_envelopes_to(node_port);
            node_port.send_processed_envelopes_to(target_port);
            return;
        }

        // TODO: implement all the possible link types

        assert(false && "Given link type is unknown");
    }

public:
    Link& write(Device::index_t device_index) {
        return this->create_link(device_index, hub::LinkFlags::write);
    }

    Link& read(Device::index_t device_index) {
        return this->create_link(device_index, hub::LinkFlags::read);
    }

    // TODO: readwrite(...), read_peek(...), readwrite_peek(...)
};

} // pipelines namespace
} // namespace noarr

#endif

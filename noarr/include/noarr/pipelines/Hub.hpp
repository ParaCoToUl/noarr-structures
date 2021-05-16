#ifndef NOARR_PIPELINES_HUB_HPP
#define NOARR_PIPELINES_HUB_HPP

#include <vector>

#include "UntypedPort.hpp"
#include "Device.hpp"
#include "Node.hpp"

#include "Hub_Link.hpp"
#include "Hub_Chunk.hpp"
#include "Hub_BufferPool.hpp"
#include "Hub_ChunkQueue.hpp"

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
class Hub : public Node {
public:
    using Link = hub::Link<Structure, BufferItem>;
    using Chunk = hub::Chunk<Structure, BufferItem>;

    /**
     * Holds empty buffers and distributes them to write links
     */
    hub::BufferPool<Structure, BufferItem> buffer_pool;

    /**
     * Holds freshly produced envelopes, as they arrive into the hub
     * TODO: merge with the DistributeQueue and rename to it
     */
    hub::ChunkQueue<Structure, BufferItem> chunk_queue;

    // TODO: construction API to be figured out
    Hub() : Node(typeid(Hub).name()), buffer_pool() { }

    //////////////
    // Node API //
    //////////////

    virtual void register_ports(std::function<void(UntypedPort*)> register_port) {
        this->chunk_queue.register_ports(register_port);
    };

    bool can_advance() override {
        // check all chunk queue stages
        if (this->chunk_queue.can_accept_chunk())
            return true;

        if (this->chunk_queue.can_distribute_chunk())
            return true;
        
        if (this->chunk_queue.can_send_chunk())
            return true;
        
        // nothing to do
        return false;
    }

    void advance(std::function<void()> callback) override {
        // perform simple synchronous tasks
        this->chunk_queue.try_accept_chunk();
        this->chunk_queue.try_send_chunk();

        // try to start asynchronous operations
        if (this->chunk_queue.can_distribute_chunk()) {
            this->chunk_queue.distribute_chunk(callback);
            return;
        }

        // computation is done, since no asynchronous operation was launched
        callback();
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
            auto& source_port = this->buffer_pool.create_output_port(dev);
            auto& target_port = this->chunk_queue.create_input_port(dev);
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

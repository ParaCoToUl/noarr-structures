#ifndef NOARR_PIPELINES_HUB_HPP
#define NOARR_PIPELINES_HUB_HPP

#include <vector>
#include <map>
#include <deque>
#include <iostream>

#include "Device.hpp"
#include "Node.hpp"
#include "Link.hpp"
#include "Buffer.hpp"
#include "HardwareManager.hpp"

#include "Hub_Chunk.hpp"

namespace noarr {
namespace pipelines {

/**
 * Hub is responsible for buffer allocation and transfer. It manages a set
 * of envelopes and provides these envelopes to the other nodes via links.
 */
template<typename Structure, typename BufferItem>
class Hub : public Node {
private:
    using Envelope_t = Envelope<Structure, BufferItem>;
    using Chunk_t = hub::Chunk<Structure, BufferItem>;
    using Link_t = Link<Structure, BufferItem>;

    HardwareManager& hardware_manager;

    /**
     * Size of all envelopes in this hub in bytes
     * (buffer sizes, so the capacity)
     */
    std::size_t buffer_size;

    /**
     * List of all envelopes in this hub
     * (for memory management)
     */
    std::vector<std::unique_ptr<Envelope_t>> envelopes;

    /**
     * List of all chunks in this hub
     * (for memory management)
     */
    std::vector<std::unique_ptr<Chunk_t>> chunks;

    /**
     * List of all links this hub hosts
     */
    std::vector<std::unique_ptr<Link_t>> links;

    /**
     * Empty envelopes, available to be used
     */
    std::map<Device::index_t, std::vector<Envelope_t*>> empty_envelopes;

    /**
     * Chunks that have been produced and are now waiting to be peeked and consumed
     */
    std::deque<Chunk_t*> chunk_queue;

    /**
     * Chunks that have been consumed from the queue, but are still being peeked
     * and so cannot be truly freed up
     */
    std::vector<Chunk_t*> consumed_chunks;

public:
    Hub(std::size_t buffer_size)
        : Hub(buffer_size, HardwareManager::default_manager())
    { }

    Hub(std::size_t buffer_size, HardwareManager& hardware_manager) :
        Node(typeid(Hub).name()),
        hardware_manager(hardware_manager),
        buffer_size(buffer_size)
    { }

    /**
     * Allocates new envelopes on a given device
     */
    void allocate_envelopes(Device::index_t device_index, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i)
            this->allocate_envelope(device_index);
    }

    /**
     * Allocates a new envelope on the given device
     */
    void allocate_envelope(Device::index_t device_index) {
        envelopes.push_back(
            std::make_unique<Envelope_t>(
                hardware_manager.allocate_buffer(device_index, buffer_size)
            )
        );
        
        empty_envelopes[device_index].push_back(&*envelopes.back());
    }

    /**
     * Called synchronously during initialization to initialize the hub content
     */
    Envelope_t& push_new_chunk() {
        if (empty_envelopes[Device::HOST_INDEX].empty()) {
            assert(false && "No empty envelope available on the host");
        }

        Envelope_t& envelope = *empty_envelopes[Device::HOST_INDEX].back();
        empty_envelopes[Device::HOST_INDEX].pop_back();
        
        chunks.push_back(std::make_unique<Chunk_t>(envelope, Device::HOST_INDEX));
        chunk_queue.push_back(&*chunks.back());

        return envelope;
    }

    /**
     * Called synchronously during finalization to access the latest chunk
     */
    Envelope_t& peek_latest_chunk() {
        if (chunk_queue.empty()) {
            assert(false && "The hub contains no chunks");
        }

        Chunk_t& chunk = *chunk_queue.front();

        if (chunk.envelopes.find(Device::HOST_INDEX) == chunk.envelopes.end()) {
            // TODO: do a synchronous copy, maybe?
            assert(false && "The latest chunk is not present on the host");
        }

        return *chunk.envelopes[Device::HOST_INDEX];
    }

    /**
     * Removes the latest chunk from the chunk queue
     */
    void consume_latest_chunk() {
        consumed_chunks.push_back(chunk_queue.back());
        chunk_queue.pop_back();
    }

    //////////////
    // Node API //
    //////////////

public:

    bool can_advance() override {
        if (put_empty_envelope_to_producing_link(true))
            return true;

        if (link_latest_chunk_for_peeks_and_modifications(true))
            return true;
        
        // nothing to do
        return false;
    }

    void advance() override {
        // perform simple synchronous tasks
        put_empty_envelope_to_producing_link();
        link_latest_chunk_for_peeks_and_modifications();

        // // try to start asynchronous operations
        // if (this->chunk_queue.can_distribute_chunk()) {
        //     this->chunk_queue.distribute_chunk(callback);
        //     return;
        // }

        // computation is done, since no asynchronous operation was launched
        this->callback();
    }

private:

    bool put_empty_envelope_to_producing_link(bool dry_run = false) {
        for (std::unique_ptr<Link_t>& link_ptr : links) {
            Link_t& link = *link_ptr;

            if (link.type != LinkType::producing)
                continue;

            if (link.envelope != nullptr)
                continue;

            if (empty_envelopes.find(link.device_index) != empty_envelopes.end()) {
                // found an empty producing link and an empty envelope
                if (!dry_run) {
                    link.host_envelope(
                        *empty_envelopes[link.device_index].back(),
                        [this, &link](){
                            finish_producing_link(link);
                        }
                    );
                    empty_envelopes[link.device_index].pop_back();
                }
                return true;
            }
        }

        return false;
    }

    void finish_producing_link(Link_t& link) {
        if (link.was_committed) {
            chunks.push_back(std::make_unique<Chunk_t>(*link.envelope, link.device_index));
            chunk_queue.push_back(&*chunks.back());
        } else {
            empty_envelopes[link.device_index].push_back(&*envelopes.back());
        }

        link.detach_envelope();
    }

    bool link_latest_chunk_for_peeks_and_modifications(bool dry_run = false) {
        if (chunk_queue.size() == 0)
            return false;

        Chunk_t& chunk = *chunk_queue.front();
        
        bool ret = false;

        for (std::unique_ptr<Link_t>& link_ptr : links) {
            Link_t& link = *link_ptr;

            if (link.type != LinkType::peeking && link.type != LinkType::modifying)
                continue;

            if (link.envelope != nullptr)
                continue;

            if (chunk.envelopes.find(link.device_index) != chunk.envelopes.end()) {
                if (!dry_run) {
                    link.host_envelope(
                        *chunk.envelopes[link.device_index],
                        [this, &link, &chunk](){
                            finish_peek_or_modification_link(link, chunk);
                        }
                    );

                    if (link.type == LinkType::peeking)
                        chunk.peeking_count += 1;

                    if (link.type == LinkType::modifying)
                        chunk.modifying_count += 1;
                }
                ret = true;
            }
        }

        return ret;
    }

    void finish_peek_or_modification_link(Link_t& link, Chunk_t& chunk) {
        if (link.type == LinkType::peeking)
            chunk.peeking_count -= 1;
        
        if (link.type == LinkType::modifying)
            chunk.modifying_count -= 1;

        link.detach_envelope();
    }

    ///////////////////////
    // Link creation API //
    ///////////////////////

public:

    /**
     * Creates a new producing link
     */
    Link_t& to_produce(Device::index_t device_index, bool autocommit = true) {
        return create_link(LinkType::producing, device_index, autocommit);
    }

    /**
     * Creates a new consuming link
     */
    Link_t& to_consume(Device::index_t device_index, bool autocommit = true) {
        return create_link(LinkType::consuming, device_index, autocommit);
    }

    /**
     * Creates a new peeking link
     */
    Link_t& to_peek(Device::index_t device_index) {
        return create_link(LinkType::peeking, device_index, false);
    }

    /**
     * Creates a new modifying link
     */
    Link_t& to_modify(Device::index_t device_index) {
        return create_link(LinkType::modifying, device_index, false);
    }

    /**
     * Creates a new link for which this hub is the host
     */
    Link_t& create_link(LinkType type, Device::index_t device_index, bool autocommit) {
        links.push_back(std::make_unique<Link_t>(
            type,
            device_index,
            autocommit
        ));
        return *links.back();
    }
};

} // pipelines namespace
} // namespace noarr

#endif

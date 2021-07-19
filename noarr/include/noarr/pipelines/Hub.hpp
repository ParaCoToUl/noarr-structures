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

    /**
     * Hardware manager used for allocations and memory transfers
     */
    HardwareManager& hardware_manager;

    // poitners to envelopes participating in a memory transfer
    Envelope_t* source_envelope_for_transfer = nullptr;
    Envelope_t* target_envelope_for_transfer = nullptr;

    /**
     * Size of all envelopes in this hub in bytes
     * (buffer sizes, so the capacity)
     */
    std::size_t buffer_size;

    /**
     * Maximum length the queue can be.
     * When set to zero, it's considered to be without limit.
     * (limited only by the number of available envelopes)
     * 
     * Setting a limit is useful for preventin hub from exhausting all available
     * envelopes on a given device and thus not having envelopes
     * for memory transfers.
     */
    std::size_t max_queue_length = 0;

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
     * Contains envelopes that have been thrown away as not useful anymore,
     * but they might still be used by some links and so cannot be immediately
     * reused as empty envelopes.
     */
    std::vector<Envelope_t*> trashed_envelopes;

    /**
     * Chunks that have been produced and are now waiting to be peeked and consumed
     * 
     * Index [0] is the consuming end and index [size - 1] is the producing end.
     * Adding new chunk is "push_back" and consuming next chunk is "pop_front".
     * 
     * The chunk that's about to be consumed is the "top chunk" (at index [0]).
     */
    std::deque<Chunk_t*> chunk_queue;

    /**
     * Links, to which data should flow
     */
    std::vector<Link_t*> dataflow_links;

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

        say("Allocated new envelope on device: " + std::to_string(device_index));
    }

    /**
     * Sets the maximum allowed queue length
     */
    void set_max_queue_length(std::size_t max_length) {
        max_queue_length = max_length;
    }

    /**
     * Returns the current length of the queue
     */
    std::size_t get_queue_length() const {
        return chunk_queue.size();
    }

    /**
     * Makes the dataflow empty.
     * 
     * This prevents consumtion so make sure to combine it with other dataflow
     * setting methods.
     */
    void reset_dataflow() {
        dataflow_links.clear();
    }

    /**
     * Sets the dataflow to point to a given node or link
     */
    void flow_data_to(Node& node, bool reset_flow_first = true) {
        Link_t& link = resolve_link_from_node(node);
        flow_data_to(link, reset_flow_first);
    }

    /**
     * Sets the dataflow to point to a given node or link
     */
    void flow_data_to(Link_t& link, bool reset_flow_first = true) {
        if (reset_flow_first)
            reset_dataflow();

        // check that the link isn't producing (how could we flow data into that?)
        assert(
            link.type != LinkType::producing
            && "You cannot flow data into a producing link."
        );

        // check that is we have a modifying link, all other existing modifying
        // links have to be on the same device (otherwise how do we merge them?)
        if (link.type == LinkType::modifying) {
            for (Link_t* link_ptr : dataflow_links) {
                Link_t& dataflow_link = *link_ptr;

                if (dataflow_link.type == LinkType::modifying) {
                    assert(
                        dataflow_link.device_index == link.device_index
                        && "Dataflow with multiple modifying links must have all of them on the same device."
                    );
                }
            }
        }
        
        // accept the link into the dataflow
        dataflow_links.push_back(&link);
    }

    /**
     * Sets the dataflow to also point to a given node or link
     */
    void and_flow_data_to(Node& node) {
        flow_data_to(node, false);
    }

    /**
     * Sets the dataflow to also point to a given node or link
     */
    void and_flow_data_to(Link_t& link) {
        flow_data_to(link, false);
    }

    /**
     * Attempts to automatically infer dataflow, returns true on success
     */
    bool infer_dataflow() {
        // TLDR: Get all non-producing links and if there's only one such link,
        // use that. Otherwise there's none, or we cannot decide on which to use,
        // so don't infer anything.

        bool found_a_link = false;

        for (std::unique_ptr<Link_t>& link_ptr : links) {
            Link_t& link = *link_ptr;

            // if we have a non-producing link
            if (link.type == LinkType::consuming
                || link.type == LinkType::modifying
                || link.type == LinkType::peeking)
            {
                // we've already found one, we cannot decide on which one to use
                if (found_a_link)
                    return false;

                // now we have the link
                flow_data_to(link);
                found_a_link = true;

                // don't return yet, we need to check for ambiguity
            }
        }

        return found_a_link;
    }

    /**
     * Can be called synchronously during initialization to initialize the hub content
     * 
     * The returned reference has only short lifetime, don't use it outside the
     * initialization method
     */
    Envelope_t& push_new_chunk() {
        if (empty_envelopes[Device::HOST_INDEX].empty()) {
            assert(false && "No empty envelope available on the host");
        }

        // check the queue length constraint
        if (max_queue_length > 0) {
            if (chunk_queue.size() >= max_queue_length)
                assert(false && "Queue is already at its size limit");
        }

        Envelope_t& envelope = *empty_envelopes[Device::HOST_INDEX].back();
        empty_envelopes[Device::HOST_INDEX].pop_back();
        
        chunks.push_back(std::make_unique<Chunk_t>(envelope, Device::HOST_INDEX));
        chunk_queue.push_back(&*chunks.back());

        return envelope;
    }

    /**
     * Can be called synchronously during finalization to read the top chunk
     * 
     * The returned reference has only short lifetime, don't use it outside the
     * finalization method
     */
    Envelope_t& peek_top_chunk() {
        if (chunk_queue.empty()) {
            assert(false && "The hub contains no chunks");
        }

        // get the top chunk
        Chunk_t& top_chunk = *chunk_queue.front();

        // check that it's present on the host
        if (top_chunk.envelopes.count(Device::HOST_INDEX) == 0) {
            // TODO: do a synchronous copy, maybe?
            assert(false && "The latest chunk is not present on the host");
        }

        // return reference to the envelope
        return *top_chunk.envelopes[Device::HOST_INDEX];
    }

    /**
     * Removes the top chunk from the queue
     */
    void consume_top_chunk() {
        if (chunk_queue.empty()) {
            assert(false && "The hub contains no chunks");
        }

        // get the top chunk
        Chunk_t& top_chunk = *chunk_queue.front();

        // transfer its envelopes into trash
        for (auto const& x : top_chunk.envelopes) {
            trashed_envelopes.push_back(x.second);
        }

        // remove the chunk from the queue
        chunk_queue.pop_front();

        say("Top chunk consumed.");
    }

    /////////////
    // Logging //
    /////////////

private:

    std::ostream* log_stream_ptr = nullptr;

    void say(const std::string& message) {
        if (log_stream_ptr == nullptr)
            return;

        *log_stream_ptr << "[" << label << "]: " << message << std::endl;
    }

public:

    /**
     * Enables logging of hub events to a given stream for debugging
     */
    void start_logging(std::ostream& output_stream) {
        log_stream_ptr = &output_stream;
    }

    //////////////
    // Node API //
    //////////////

public:

    void initialize() override {
        // try to infer dataflow if the user didn't specify an explicit one
        if (dataflow_links.empty() && !infer_dataflow())
            assert(false && "Dataflow is empty and cannot be infered. Define an explicit one.");
    }

    bool can_advance() override {
        // clean up the trash
        // (does have side-effects, but is necessary for the following dry-runs
        // to run correctly)
        transfer_trashed_envelopes_to_empty_envelopes();

        if (host_producing_links(true))
            return true;

        if (host_top_chunk_in_dataflow_links(true))
            return true;

        if (start_memory_transfer(true))
            return true;
        
        // nothing to do
        return false;
    }

    void advance() override {
        // perform simple synchronous tasks
        host_producing_links();
        host_top_chunk_in_dataflow_links();

        // try to start one asynchronous operation (data transfer)
        // (doesn't run multiple async operations due to the complexity of
        // managing callbacks in such case)
        if (start_memory_transfer())
            return; // if the transfer does start, it will handle callback calling

        // computation is done, since no asynchronous operation was launched
        this->callback();
    }

    void post_advance() override {
        // if there was a memory transfer, now finalize it
        if (source_envelope_for_transfer != nullptr
            || target_envelope_for_transfer != nullptr)
        {
            source_envelope_for_transfer = nullptr;
            target_envelope_for_transfer = nullptr;

            // clean up the trash
            transfer_trashed_envelopes_to_empty_envelopes();
        }
    }

private:

    /**
     * Goes through trashed envelopes and transfers those that are no longer
     * used into empty envelopes
     */
    void transfer_trashed_envelopes_to_empty_envelopes() {
        for (std::size_t i = 0; i < trashed_envelopes.size(); ++i) {
            Envelope_t& envelope = *trashed_envelopes[i];

            // skip envelopes involved in memory transfer
            if (source_envelope_for_transfer == &envelope) continue;
            if (target_envelope_for_transfer == &envelope) continue;

            // skip envelopes hosted in links
            if (is_envelope_hosted_in_a_link(envelope))
                continue;

            // :: we have an unused envelope ::

            // remove envelope from the trash
            trashed_envelopes.erase(trashed_envelopes.begin() + i);
            --i;

            // put the envelope into empty envelopes
            empty_envelopes[envelope.device_index].push_back(&envelope);

            say(
                "Recovered an empty envelope for device: "
                + std::to_string(envelope.device_index)
            );
        }
    }

    /**
     * Returns true if there is a link that currently hosts given envelope
     */
    bool is_envelope_hosted_in_a_link(Envelope_t& envelope) {
        // go over all links
        for (std::unique_ptr<Link_t>& link_ptr : links) {
            Link_t& link = *link_ptr;

            if (link.envelope == &envelope)
                return true; // yep, it's hosted here
        }

        return false; // no it isn't hosted anywhere
    }

    /**
     * Tries to put empty envelopes into producing links
     */
    bool host_producing_links(bool dry_run = false) {
        // check the queue length constraint
        if (max_queue_length > 0) {
            if (chunk_queue.size() >= max_queue_length)
                return false; // cannot host anything, queue too long
        }
        
        // go over all producing links
        for (std::unique_ptr<Link_t>& link_ptr : links) {
            Link_t& link = *link_ptr;

            // go over producing links only
            if (link.type != LinkType::producing)
                continue;

            // skip links that are already occupied
            if (link.envelope != nullptr)
                continue;

            // there has to be an empty envelope on the device for the link
            if (empty_envelopes[link.device_index].size() == 0)
                continue;

            // found an empty producing link and an empty envelope
            if (!dry_run) {
                say("Hosting producing link to node: " + link.guest_node->label);

                // host the envelope
                link.host_envelope(
                    *empty_envelopes[link.device_index].back(),
                    [this, &link](){
                        finish_producing_link_hosting(link);
                    }
                );

                // and the envelope is no longer empty
                empty_envelopes[link.device_index].pop_back();
            }
            return true;
        }

        // no links to be hosted
        return false;
    }

    void finish_producing_link_hosting(Link_t& link) {
        // the production did happen, the envelope in the link is full of data
        if (link.was_committed) {
            // create a new chunk and push it into the queue
            chunks.push_back(std::make_unique<Chunk_t>(*link.envelope, link.device_index));
            chunk_queue.push_back(&*chunks.back());
            
            say("New chunk produced by: " + link.guest_node->label);
        }
        
        // the production didn't happen, the envelope in the link remains empty
        else {
            // return the envelope back to empty envelopes
            empty_envelopes[link.device_index].push_back(link.envelope);

            say("Chunk production wasn't commited, by: " + link.guest_node->label);
        }

        // take the envelope out of the link
        link.detach_envelope();
    }

    /**
     * Tries to host the top chunk in current dataflow links
     */
    bool host_top_chunk_in_dataflow_links(bool dry_run = false) {
        if (chunk_queue.size() == 0)
            return false;

        // obtain the top chunk
        Chunk_t& top_chunk = *chunk_queue.front();
        
        bool ret = false;

        // go through all dataflow links and try to host the chunk on each of them
        for (Link_t* link_ptr : dataflow_links) {
            Link_t& link = *link_ptr;

            // skip links that are already occupied
            if (link.envelope != nullptr)
                continue;

            // skip links that are on devices where the chunk hasn't been transfered yet
            if (top_chunk.envelopes.count(link.device_index) == 0)
                continue;
            
            if (!dry_run) {
                say("Hosting top chunk to node: " + link.guest_node->label);
                
                // host the chunk
                link.host_envelope(
                    *top_chunk.envelopes[link.device_index],
                    [this, &link, &top_chunk](){
                        finish_top_chunk_hosting(link, top_chunk);
                    }
                );
            }
            ret = true;
        }

        return ret;
    }

    void finish_top_chunk_hosting(Link_t& link, Chunk_t& top_chunk) {
        // the consumption or modification did happen
        if (link.was_committed) {
            // the chunk was consumed
            if (link.type == LinkType::consuming) {
                // pop the chunk off the queue
                consume_top_chunk();
            }

            // the chunk was modified
            if (link.type == LinkType::modifying) {
                // trash all envelopes except the one that was modified
                for (auto const& x : top_chunk.envelopes) {
                    if (x.first == link.device_index)
                        continue; // don't trash the modified envelope
                    trashed_envelopes.push_back(x.second);
                }

                // rebuild the envelope set for the chunk
                // (so that is contains only the one modified)
                top_chunk.envelopes.clear();
                top_chunk.envelopes[link.device_index] = link.envelope;

                say("Top chunk was modified by node: " + link.guest_node->label);
            }
        }

        else {
            // Do nothing - not consumed, nor modified, so maybe was read.
            // The chunk should stay where it is.

            say(
                "Top chunk hosting finished without commitment. To node: "
                + link.guest_node->label
            );
        }

        // take the envelope out of the link
        link.detach_envelope();

        // clean up the trash
        transfer_trashed_envelopes_to_empty_envelopes();
    }

    /**
     * Tries to start one memory transfer and if it succeeds, it returns true
     * When the transfer finishes, it will call this->callback()
     */
    bool start_memory_transfer(bool dry_run = false) {
        // go over all chunks, from the top chunk to the newer chunks
        for (Chunk_t* chunk_ptr : chunk_queue) {
            Chunk_t& chunk = *chunk_ptr;
            
            // and go over all dataflow links
            for (Link_t* link_ptr : dataflow_links) {
                Link_t& link = *link_ptr;
                Device::index_t target_device_index = link.device_index;

                // skip devices, where the chunk is already present
                if (chunk.envelopes.count(target_device_index) == 1)
                    continue;

                // if it's not present, we can start a transfer
                if (!dry_run) {
                    say("Transferring chunk to the device of node: " + link.guest_node->label);

                    // get an envelope to which we'll copy the data
                    Envelope_t& target_env = obtain_envelope_for_transfer(
                        target_device_index
                    );

                    // get an envelope from which we'll copy the data
                    Envelope_t& source_env = chunk.get_source_for_transfer(
                        target_device_index
                    );

                    // remember participating envelopes so that they
                    // don't get freed up
                    source_envelope_for_transfer = &source_env;
                    target_envelope_for_transfer = &target_env;

                    // copy structure
                    target_env.structure = source_env.structure;

                    // copy data
                    hardware_manager.transfer_data(
                        source_env.allocated_buffer_instance,
                        target_env.allocated_buffer_instance,
                        // FUTURE FEATURE:
                        // an optimization is to calculate transferred size
                        // from the source envelope structure
                        buffer_size, // <---
                        [this, &chunk, target_device_index, &target_env](){
                            // put the envelope into the chunk
                            chunk.envelopes[target_device_index] = &target_env;

                            // finish node execution
                            this->callback();

                            // follows a bit of code in the post_advance method
                        }
                    );
                }
                return true;
            }
        }

        // no memory transfer was started
        return false;
    }

    /**
     * Obtains an empty envelope for data transfer onto a device
     */
    Envelope_t& obtain_envelope_for_transfer(Device::index_t target) {
        assert(
            !empty_envelopes[target].empty()
            && "There are no available envelopes for memory transfer"
        );

        Envelope_t& envelope = *empty_envelopes[target].back();
        empty_envelopes[target].pop_back();

        return envelope;
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
    Link_t& to_modify(Device::index_t device_index, bool autocommit = true) {
        return create_link(LinkType::modifying, device_index, autocommit);
    }

    /**
     * Creates a new link for which this hub is the host
     */
    Link_t& create_link(LinkType type, Device::index_t device_index, bool autocommit) {
        links.push_back(std::make_unique<Link_t>(
            this,
            type,
            device_index,
            autocommit
        ));
        return *links.back();
    }

    /**
     * Tries to find a link that points to given node and fails
     * if there's none or multiple links
     */
    Link_t& resolve_link_from_node(Node& node) {
        Link_t* resolved_link_ptr = nullptr;
        
        for (std::unique_ptr<Link_t>& link_ptr : links) {
            Link_t& link = *link_ptr;

            // skip links pointing to different nodes
            if (link.guest_node != &node)
                continue;

            // was already resolved
            if (resolved_link_ptr != nullptr)
                assert(false && "There are multiple links to the given node");

            resolved_link_ptr = &link;
        }

        if (resolved_link_ptr == nullptr)
            assert(false && "There are no links to the given node");

        return *resolved_link_ptr;
    }
};

} // pipelines namespace
} // namespace noarr

#endif

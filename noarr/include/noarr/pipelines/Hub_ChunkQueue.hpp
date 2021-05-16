#ifndef NOARR_PIPELINES_HUB_CHUNK_QUEUE_HPP
#define NOARR_PIPELINES_HUB_CHUNK_QUEUE_HPP

#include <map>
#include <vector>
#include <deque>

#include "Device.hpp"
#include "Port.hpp"
#include "Envelope.hpp"

#include "Hub_Chunk.hpp"

namespace noarr {
namespace pipelines {
namespace hub {

/**
 * Implements the chunk queue of a hub
 */
template<typename Structure, typename BufferItem = void>
class ChunkQueue {
public:
    using Chunk_t = Chunk<Structure, BufferItem>;
    using Port_t = Port<Structure, BufferItem>;
    using Envelope_t = Envelope<Structure, BufferItem>;

    /**
     * Ports (one for each producer) where freshly filled envelopes are received
     */
    std::vector<Port_t> input_ports;

    /**
     * Ports (one for each device) where envelopes leave the queue
     */
    std::map<Device::index_t, Port_t> output_ports;

    /**
     * The actual queue of chunks
     */
    std::deque<Chunk_t> chunks;

    ChunkQueue() { }

    void register_ports(std::function<void(UntypedPort*)> register_port) {
        for (auto& p : this->input_ports)
            register_port(&p);
        for (auto& kv : this->output_ports)
            register_port(&kv.second);
    }

    /**
     * Creates a new port to which freshly filled envelopes can be sent
     */
    Port_t& create_input_port(Device::index_t device_index) {
        this->input_ports.push_back(Port_t(device_index));
        return this->input_ports.back();
    }

    /*
        Following methods implement stages through which each chunk flows
        through the queue.

        1) Chunks are injested on input ports. There's one such port for every
        node that requests to write to this hub.
        (typical usecase: only one or none producers)

        2) Chunks get distributed onto all the devices that will peek or read
        the chunk. This might include removal of the chunk from the writing
        device if it's not needed there in later stages.
        (typical usecase: move chunk from one device to another)

        3) Chunks get peeked by each link that requests so. Peeking on some
        devices can be overlapped with distribution to other devices, but
        from the perspective of one device must be the chunk order preserved.

        4) Chunks that have been peeked by everyone can be consumed. The chunk
        will be sent to the first available consuming link. So the consuming
        is competing.
        (typical usecase: only one or none consumers)
     */

    //////////////////////////////////
    // Stage 1: New chunk accepting //
    //////////////////////////////////

private:
    Port_t* _accept_chunk_port = nullptr;

public:
    bool can_accept_chunk() {
        for (Port_t& p : this->input_ports) {
            if (p.state() == PortState::arrived) {
                this->_accept_chunk_port = &p;
                return true;
            }
        }
        
        return false;
    }

    void try_accept_chunk() {
        if (!this->can_accept_chunk())
            return;
        
        Envelope_t& env = this->_accept_chunk_port->envelope();
        this->_accept_chunk_port->detach_envelope();

        // create a new chunk, give it the envelope and push it into the queue
        this->chunks.push_back(
            Chunk_t(env, this->_accept_chunk_port->device_index())
        );
    }

    ////////////////////////////////////////
    // Stage 2: Distribution onto devices //
    ////////////////////////////////////////

public:
    bool can_distribute_chunk() {
        return false;
    }

    void distribute_chunk(std::function<void()> callback) {
        // ...
        callback();
    }

private:
    bool chunk_is_distributed_completely(const Chunk_t& chunk) {
        return true; // TODO
    }

    //////////////////////
    // Stage 3: Peeking //
    //////////////////////

    // TODO ...

private:
    bool chunk_was_peeked_by_everyone(const Chunk_t& chunk) {
        return true; // TODO
    }

    //////////////////////////////
    // Stage 4: Chunk consuming //
    //////////////////////////////

private:

public:
    bool can_send_chunk() {
        // for (auto& kv : this->output_ports) {
        //     kv.first
        //     &kv.second;
        // }
        
        Chunk_t& chunk = this->chunks.front();

        return !this->chunks.empty()
            && this->chunk_is_distributed_completely(chunk)
            && this->chunk_was_peeked_by_everyone(chunk);
    }

    void try_send_chunk() {
        if (!this->can_send_chunk())
            return;

        Chunk_t& chunk = this->chunks.front();
        Envelope_t& env = chunk.envelopes;
        this->chunks.pop_front();
    }
};

} // hub namespace
} // pipelines namespace
} // namespace noarr

#endif

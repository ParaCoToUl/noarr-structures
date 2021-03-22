#ifndef NOARR_PIPELINES_ENVELOPE_HPP
#define NOARR_PIPELINES_ENVELOPE_HPP 1

#include <cstddef>
#include "noarr/pipelines/chunk_stream_processor.hpp"

namespace noarr {
namespace pipelines {

/**
 * Envelopes are responsible for moving data between buffers
 * 
 * NOTE: A general envelope is not an async worker, only some envelopes are.
 */
class envelope {

    // TODO: What do all envelopes have in common?
    // Do they need a shared interface?
    // ...
    // probably yes - an allocator, memory management, etc...

public:

    /**
     * Port is the thing that's given to a compute node for buffer resolution
     */
    template<typename Structure, typename BufferItem = void>
    class port {
    public:
        // TODO: add reference to an envelope instance, implement this

        /**
         * Returns pointer to the buffer that holds the data
         */
        BufferItem* get_buffer() {
            // TODO
            return nullptr;
        };

        /**
         * Returns structure of the data buffer
         */
        Structure get_structure() {
            // TODO
        };

        /**
         * Returns true if the port contains a valid chunk data,
         * false if it doesn't contain anything useful (is empty)
         */
        bool contains_chunk() {
            return false; // TODO: implement this
        }

        /**
         * Returns true if the port contains the EOS flag
         * (end of stream - it means no more chunks will ever be available)
         */
        bool contains_end_of_stream() {
            return false; // TODO: implement this
        }

        /**
         * Sets the contains_chunk flag
         * 
         * (to true when someone populates the buffer and to false
         * when someone consumes the buffer)
         */
        void set_contains_chunk(bool value) {
            // TODO: implement this
        }

        /**
         * Sends the EOS flag into the envelope and the envelope
         * has the ability to pass it further down the pipeline
         */
        void send_end_of_stream() {
            // TODO: implement this
        }
    };
};

/**
 * Buffer envelope holds one buffer that might, for example,
 * be used exclusively by a single compute node.
 * (it does not support the chunk_stream_processor interface, obviously)
 */
template<typename Structure, typename BufferItem = void>
class buffer_envelope : public envelope {
public:
    /**
     * Returns a port that provides access to the underlying buffer
     */
    envelope::port<Structure, BufferItem> get_port() {
        // TODO: implement this
    }
};

/**
 * Pipe envelope has one input port and one output port
 */
template<
    typename InputStructure,
    typename OutputStructure,
    typename InputBufferItem = void,
    typename OutputBufferItem = void
>
class pipe_envelope : public chunk_stream_processor {
    /**
     * Returns the input port for for this pipeline envelope
     */
    virtual envelope::port<InputStructure, InputBufferItem> get_input_port() = 0;

    /**
     * Returns the output port for for this pipeline envelope
     */
    virtual envelope::port<OutputStructure, OutputBufferItem> get_output_port() = 0;
};

class move_h2d_envelope : public envelope {
public:
    /*
        Envelope for passing data from host to device
        (maybe could be implemented  by pipe_envelope with proper settings?)
    */
};

/**
 * Represents a node that performs some computation
 * (either on device or even on the host)
 * 
 * NOTE: Compared to envelopes, all compute nodes are async workers
 */
class compute_node : public chunk_stream_processor {
    // ...
};

/**
 * A compute node with one input port and one output port
 */
template<
    typename InputStructure,
    typename OutputStructure,
    typename InputBufferItem = void,
    typename OutputBufferItem = void
>
class pipe_compute_node : public compute_node {
public:
    using input_port_t = envelope::port<InputStructure, InputBufferItem>;
    using output_port_t = envelope::port<OutputStructure, OutputBufferItem>;

    void set_input_port(input_port_t p) {
        this->input_port = p;
    }
    void set_output_port(output_port_t p) {
        this->output_port = p;
    };

protected:
    input_port_t input_port;
    output_port_t output_port;
};

/**
 * And example mapping compute node
 * (runs synchronously on the host)
 */
class my_mapping_node : public pipe_compute_node<
    std::size_t, std::size_t, int, int
> {

    bool is_ready_for_next_chunk() override {
        return this->input_port.contains_chunk();
    }

    virtual void start_next_chunk_processing() override {
        // === check end of stream ===
        if (this->input_port.contains_end_of_stream())
        {
            this->set_all_chunks_processed(); // we're done as a node
            this->output_port.send_end_of_stream(); // pass EOS down
            return;
        }

        // === perform the mapping operation ===

        // NOTE/TODO: here will be cast to a "bag" type
        int* input_buffer = (int*) this->input_port.get_buffer();
        int* output_buffer = (int*) this->output_port.get_buffer();

        // copy the number of items
        int item_count = input_buffer[0];
        output_buffer[0] = item_count;

        // perform the map operation on the values
        for (int i = 0; i < item_count; i++)
            output_buffer[1 + i] = input_buffer[1 + i] * 2; // map = *2

        // the input buffer was consumed and the output buffer
        // was filled with a chunk of data
        this->input_port.set_contains_chunk(false);
        this->output_port.set_contains_chunk(true);

        // the asynchronous operation has finished
        // (well, it wasn't asnychronous at all)
        this->set_chunk_processing_finished();
    }

};

class my_printing_node : public compute_node {
public:

    // TODO: add producer_node + consumer_node
    // TODO: maybe make pipe_node as both consumer and producer?

    bool is_ready_for_next_chunk() override {
        return this->input_port.contains_chunk();
    }

    virtual void start_next_chunk_processing() override {
        // === check end of stream ===
        if (this->input_port.has_end_of_stream())
        {
            this->output_port.send_end_of_stream();
            return;
        }
    }
};


} // pipelines namespace
} // namespace noarr

#endif

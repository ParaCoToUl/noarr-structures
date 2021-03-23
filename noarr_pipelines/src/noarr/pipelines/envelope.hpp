#ifndef NOARR_PIPELINES_ENVELOPE_HPP
#define NOARR_PIPELINES_ENVELOPE_HPP 1

#include <cstddef>
#include "noarr/pipelines/chunk_stream_processor.hpp"

namespace noarr {
namespace pipelines {

/**
 * Envelopes are responsible for moving data between buffers
 * 
 * NOTE: A general envelope is not a chunk stream processor, only some envelopes are.
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
        }

        /**
         * Returns structure of the data buffer
         */
        Structure get_structure() {
            // TODO
        }

        /**
         * Sets the structure of the data buffer
         */
        void set_structure(Structure structure) {
            // TODO
        }

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
public:
    /**
     * Returns the input port for for this pipeline envelope
     */
    virtual envelope::port<InputStructure, InputBufferItem> get_input_port() = 0;

    /**
     * Returns the output port for for this pipeline envelope
     */
    virtual envelope::port<OutputStructure, OutputBufferItem> get_output_port() = 0;
};

template<
    typename InputStructure,
    typename OutputStructure,
    typename InputBufferItem = void,
    typename OutputBufferItem = void
>
class move_h2d_envelope : public pipe_envelope<
    InputStructure, OutputStructure, InputBufferItem, OutputBufferItem
> {
public:
    /*
        Envelope for passing data from host to device
        (maybe could be implemented  by pipe_envelope with proper settings?)
     */
    move_h2d_envelope() {
        //
    }

    envelope::port<InputStructure, InputBufferItem> get_input_port() override { }
    envelope::port<OutputStructure, OutputBufferItem> get_output_port() override { }

    bool is_ready_for_next_chunk() override {
        return true;
    }
};

} // pipelines namespace
} // namespace noarr

#endif

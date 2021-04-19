#ifndef NOARR_PIPELINES_UNTYPED_DOCK_HPP
#define NOARR_PIPELINES_UNTYPED_DOCK_HPP

#include <cstddef>
#include <exception>
#include "memory_device.hpp"
#include "UntypedEnvelope.hpp"

namespace noarr {
namespace pipelines {

/*
    TODO: better API

    - attach_envelope(env)
    - detach_envelope() -> env

    - attachment type -> generic, loading, unloading
        -> checks and updates envelope's payload presence flag
 */

class untyped_dock {
public:

    /**
     * Possible states of the dock
     */
    enum state : unsigned char {
        /**
         * No envelope attached
         */
        empty = 0,

        /**
         * A envelope has been attached but hasn't been processed yet
         */
        arrived = 1,

        /**
         * The envelope has been processed and is ready to leave
         */
        processed = 2
    };

    /**
     * Returns the state of the dock
     */
    state get_state() {
        if (this->attached_envelope == nullptr)
            return state::empty;
        if (this->envelope_processed)
            return state::processed;
        return state::arrived;
    }

    /**
     * Set the target port, to which processed envelopes are sent
     */
    void send_processed_envelopes_to(untyped_dock* target) {
        this->envelope_target = target;
    }

    /**
     * Returns a reference to the attached envelope
     */
    UntypedEnvelope& get_untyped_envelope() {
        if (this->attached_envelope == nullptr)
            throw std::runtime_error("Cannot get an envelope when none is attached.");

        return *this->attached_envelope;
    }

    /**
     * Perform envelope arrival to this port
     */
    void attach_envelope(UntypedEnvelope* env) {
        if (this->attached_envelope != nullptr)
            throw std::runtime_error("There's an envelope already present.");

        // TODO: overload the equality operator?
        if (env->device.device_index != this->device.device_index)
            throw std::runtime_error("The envelope belongs to a different device.");

        this->attached_envelope = env;
        this->envelope_processed = false;
    }

    /**
     * The device on which the dock exists
     */
    memory_device device;

    UntypedEnvelope* attached_envelope = nullptr;
    
    bool envelope_processed = false;

    untyped_dock* envelope_target = nullptr;
};

} // pipelines namespace
} // namespace noarr

#endif

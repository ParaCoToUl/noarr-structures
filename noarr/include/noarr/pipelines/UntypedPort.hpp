#ifndef NOARR_PIPELINES_UNTYPED_PORT_HPP
#define NOARR_PIPELINES_UNTYPED_PORT_HPP

#include <cstddef>
#include <exception>
#include "Device.hpp"
#include "UntypedEnvelope.hpp"
#include "PortState.hpp"

namespace noarr {
namespace pipelines {

/*
    TODO: better API

    - attach_envelope(env)
    - detach_envelope() -> env

    - attachment type -> generic, loading, unloading
        -> checks and updates envelope's payload presence flag
 */

// forward declaration for the reference to the parent node to compile
class Node;

class UntypedPort {
public:
    /**
     * The node that this port belongs to
     * (set during port registration from within the Node class)
     */
    Node* parent_node;

    /**
     * Returns the state of the port
     */
    PortState get_state() {
        if (this->attached_envelope == nullptr)
            return PortState::empty;
        if (this->envelope_processed)
            return PortState::processed;
        return PortState::arrived;
    }

    /**
     * Set the target port, to which processed envelopes are sent
     */
    void send_processed_envelopes_to(UntypedPort& target) {
        this->envelope_target = &target;
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
     * The device on which the port exists
     */
    Device device;

    UntypedEnvelope* attached_envelope = nullptr;
    
    bool envelope_processed = false;

    UntypedPort* envelope_target = nullptr;
};

} // pipelines namespace
} // namespace noarr

#endif

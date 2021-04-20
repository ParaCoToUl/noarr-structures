#ifndef NOARR_PIPELINES_UNTYPED_PORT_HPP
#define NOARR_PIPELINES_UNTYPED_PORT_HPP

#include <optional>
#include <exception>

#include "Device.hpp"
#include "UntypedEnvelope.hpp"
#include "PortState.hpp"

namespace noarr {
namespace pipelines {

// forward declaration for the reference to the parent node to compile
class Node;

/**
 * Abstract base class for Ports
 */
class UntypedPort {
private:
    /**
     * Envelopes from what device this port accepts
     */
    Device::index_t _device_index;

    /**
     * Pointer to the attached envelope,
     * when null, no envelope is attached to this port
     */
    UntypedEnvelope* attached_envelope = nullptr;
    
    /**
     * Is the attached envelope processed or did it just arrive?
     * (has no meaning when no envelope attached)
     */
    bool envelope_processed = false;

    /**
     * Pointer to the port, to which processed envelopes should be sent
     * by the scheduler. When null, no target exists.
     */
    UntypedPort* envelope_target = nullptr;

public:
    /**
     * The node that this port belongs to
     * (set during port registration from within the Node class)
     */
    Node* parent_node = nullptr;

    UntypedPort(Device::index_t device_index) : _device_index(device_index) { }

    /**
     * Returns the state of the port
     */
    PortState state() const {
        if (this->attached_envelope == nullptr)
            return PortState::empty;
        if (this->envelope_processed)
            return PortState::processed;
        return PortState::arrived;
    }

    /**
     * Returns true if this port has a target for processed envelopes
     */
    bool has_target() {
        return this->envelope_target != nullptr;
    }

    /**
     * Returns reference to the target port
     */
    UntypedPort& target() {
        if (this->envelope_target == nullptr)
            throw std::runtime_error("The port has no target");

        return *this->envelope_target;
    }

    /**
     * Returns the device index this port belongs to
     */
    Device::index_t device_index() const {
        return this->_device_index;
    }

    /**
     * Set the target port, to which processed envelopes are sent
     */
    void send_processed_envelopes_to(UntypedPort& target) {
        // TODO: validate type signature
        
        this->envelope_target = &target;
    }

    /**
     * Sets the state of the envelope to processed
     * (or not, if the argument is false)
     */
    void set_processed(bool value = true) {
        if (this->attached_envelope == nullptr)
            throw std::runtime_error("There's no envelope attached");
        
        this->envelope_processed = value;
    }

    /**
     * Returns a reference to the attached envelope
     */
    UntypedEnvelope& envelope() {
        if (this->attached_envelope == nullptr)
            throw std::runtime_error("There's no envelope attached");

        return *this->attached_envelope;
    }

    /**
     * Attach an envelope to this port
     * @param processed To what value should be the processed flag set
     */
    void attach_envelope(UntypedEnvelope& env, bool processed = false) {
        if (this->attached_envelope != nullptr)
            throw std::runtime_error("There's an envelope already attached");

        // check device
        if (env.device.device_index != this->_device_index)
            throw std::runtime_error("The envelope belongs to a different device.");

        // check type signature
        // TODO ...

        // attach
        this->attached_envelope = &env;
        this->envelope_processed = processed;
    }

    /**
     * Detaches and returns the attached envelope
     */
    UntypedEnvelope& detach_envelope() {
        if (this->attached_envelope == nullptr)
            throw std::runtime_error("There's no envelope to be detached");

        UntypedEnvelope* ret = this->attached_envelope;
        this->attached_envelope = nullptr;
        return *ret;
    }
};

} // pipelines namespace
} // namespace noarr

#endif

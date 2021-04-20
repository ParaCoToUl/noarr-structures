#ifndef NOARR_PIPELINES_UNTYPED_PORT_HPP
#define NOARR_PIPELINES_UNTYPED_PORT_HPP

#include <cassert>

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

    /**
     * Type of the structure value
     */
    const std::type_index structure_type;

    /**
     * Type of the buffer pointer
     */
    const std::type_index buffer_item_type;

public:
    /**
     * The node that this port belongs to
     * (set during port registration from within the Node class)
     */
    Node* parent_node = nullptr;

    UntypedPort(
        Device::index_t device_index,
        const std::type_index structure_type,
        const std::type_index buffer_item_type
    ) :
        _device_index(device_index),
        structure_type(structure_type),
        buffer_item_type(buffer_item_type) { }

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
        assert(this->envelope_target != nullptr
            && "The port has no target");

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
        assert(this->structure_type == target.structure_type
            && this->buffer_item_type == target.buffer_item_type
            && "Given target has different type signature");
        
        this->envelope_target = &target;
    }

    /**
     * Sets the state of the envelope to processed
     * (or not, if the argument is false)
     */
    void set_processed(bool value = true) {
        assert(this->attached_envelope != nullptr
            && "There's no envelope to attached");
        
        this->envelope_processed = value;
    }

    /**
     * Returns a reference to the attached envelope
     */
    UntypedEnvelope& envelope() {
        assert(this->attached_envelope != nullptr
            && "There's no envelope to attached");

        return *this->attached_envelope;
    }

    /**
     * Attach an envelope to this port
     * @param processed To what value should be the processed flag set
     */
    void attach_envelope(UntypedEnvelope& env, bool processed = false) {
        assert(this->attached_envelope == nullptr
            && "There's an envelope already attached");

        assert(env.device.device_index == this->_device_index
            && "The envelope belongs to a different device");

        assert(this->structure_type == env.structure_type
            && this->buffer_item_type == env.buffer_item_type
            && "The envelope has different type signature");

        // attach
        this->attached_envelope = &env;
        this->envelope_processed = processed;
    }

    /**
     * Detaches and returns the attached envelope
     */
    UntypedEnvelope& detach_envelope() {
        assert(this->attached_envelope != nullptr
            && "There's no envelope to be detached");

        UntypedEnvelope* ret = this->attached_envelope;
        this->attached_envelope = nullptr;
        return *ret;
    }
};

} // pipelines namespace
} // namespace noarr

#endif

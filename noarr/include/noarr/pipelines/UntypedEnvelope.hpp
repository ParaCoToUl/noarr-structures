#ifndef NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP
#define NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP

#include <string>
#include <typeindex>

#include "Device.hpp"

namespace noarr {
namespace pipelines {

/**
 * Abstract base class for Envelopes that allows polymorphism
 * 
 * While "untyped" it actually always has a type, only not via
 * a template here, but via runtime variables
 */
class UntypedEnvelope {
public:
    /**
     * Flag that determines whether the envelope is considered full or empty
     * TODO: this flag may be redundant, try to remove it
     */
    bool has_payload = false;

    /**
     * Pointer to the underlying buffer
     */
    void* untyped_buffer = nullptr;

    /**
     * Size of the data buffer in bytes
     */
    std::size_t size;

    /**
     * What device this envelope lives on
     */
    Device device;

    /**
     * Label that can be used in logging and error messages
     */
    std::string label;

    /**
     * Type of the structure value
     */
    const std::type_index structure_type;

    /**
     * Type of the buffer pointer
     */
    const std::type_index buffer_item_type;

    UntypedEnvelope(
        Device device,
        void* existing_buffer,
        std::size_t buffer_size,
        const std::type_index structure_type,
        const std::type_index buffer_item_type
    ) :
        untyped_buffer(existing_buffer),
        size(buffer_size),
        device(device),
        label(std::to_string((unsigned long)this)),
        structure_type(structure_type),
        buffer_item_type(buffer_item_type) { }

protected:
    // virtual method needed for polymorphism..
    // TODO: implement this class and add some virtual methods
    virtual void foo() = 0;
};

} // pipelines namespace
} // namespace noarr

#endif

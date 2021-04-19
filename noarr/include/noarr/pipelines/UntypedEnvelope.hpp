#ifndef NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP
#define NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP

#include <string>

#include "Device.hpp"

namespace noarr {
namespace pipelines {

class UntypedEnvelope {
public:
    /**
     * Flag that determines whether the envelope is considered full or empty
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

    UntypedEnvelope(
        Device device,
        void* existing_buffer,
        std::size_t buffer_size
    ) : device(device), untyped_buffer(existing_buffer), size(buffer_size),
        label(std::to_string((unsigned long)this)) { }

protected:
    // virtual method needed for polymorphism..
    // TODO: implement this class and add some virtual methods
    virtual void foo() = 0;
};

} // pipelines namespace
} // namespace noarr

#endif

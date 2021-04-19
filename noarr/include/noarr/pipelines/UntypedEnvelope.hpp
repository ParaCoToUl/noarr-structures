#ifndef NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP
#define NOARR_PIPELINES_UNTYPED_ENVELOPE_HPP

#include <cstddef>
#include "memory_device.hpp"

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
     * What device this ship lives on
     */
    memory_device device;

    UntypedEnvelope(
        memory_device device,
        void* existing_buffer,
        std::size_t buffer_size
    ) {
        this->device = device;
        this->untyped_buffer = existing_buffer;
        this->size = buffer_size;
    }

protected:
    // virtual method needed for polymorphism..
    // TODO: implement this class and add some virtual methods
    virtual void foo() = 0;
};

} // pipelines namespace
} // namespace noarr

#endif

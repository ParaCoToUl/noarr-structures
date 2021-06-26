#ifndef NOARR_PIPELINES_ENVELOPE_HPP
#define NOARR_PIPELINES_ENVELOPE_HPP

#include <cstddef>
#include "UntypedEnvelope.hpp"

namespace noarr {
namespace pipelines {

template<typename Structure, typename BufferItem = void>
class Envelope : public UntypedEnvelope {
public:
    /**
     * The structure of data contained in the envelope
     */
    Structure structure;

    /**
     * Pointer to the underlying data buffer
     */
    BufferItem* buffer;

    /**
     * Constructs a new envelope from an existing buffer
     */
    Envelope(
        Device device,
        void* existing_buffer,
        std::size_t buffer_size
    ) : UntypedEnvelope(
            device,
            existing_buffer,
            buffer_size,
            typeid(Structure),
            typeid(BufferItem)
    ) {
        this->buffer = (BufferItem*) existing_buffer;
    }

protected:
    // virtual method needed for polymorphism..
    // TODO: implement this class and add some virtual methods
    void foo() override {};
};

} // pipelines namespace
} // namespace noarr

#endif

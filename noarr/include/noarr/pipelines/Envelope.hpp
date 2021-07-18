#ifndef NOARR_PIPELINES_ENVELOPE_HPP
#define NOARR_PIPELINES_ENVELOPE_HPP

#include <cstddef>

#include "Buffer.hpp"
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
    Envelope(Buffer allocated_buffer)
        : UntypedEnvelope(
            std::move(allocated_buffer),
            typeid(Structure),
            typeid(BufferItem)
        ),
        buffer((BufferItem*) untyped_buffer)
    { }
};

} // pipelines namespace
} // namespace noarr

#endif

#ifndef NOARR_PIPELINES_HUB_CHUNK_HPP
#define NOARR_PIPELINES_HUB_CHUNK_HPP

#include <map>

#include "Device.hpp"
#include "Envelope.hpp"

namespace noarr {
namespace pipelines {
namespace hub {

/**
 * A chunk of data inside an Envelope Hub
 */
template<typename Structure, typename BufferItem = void>
class Chunk {
public:

    /**
     * All envelopes on all devices that hold the same data (this chunk)
     */
    std::map<Device::index_t, Envelope<Structure, BufferItem>&> envelopes;

    Chunk(
        Envelope<Structure, BufferItem>& envelope,
        Device::index_t device_index
    ) {
        envelopes[device_index] = envelope;
    }
};

} // hub namespace
} // pipelines namespace
} // namespace noarr

#endif

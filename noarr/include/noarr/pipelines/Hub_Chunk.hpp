#ifndef NOARR_PIPELINES_HUB_CHUNK_HPP
#define NOARR_PIPELINES_HUB_CHUNK_HPP

#include <map>

#include "Device.hpp"
#include "Envelope.hpp"

namespace noarr {
namespace pipelines {
namespace hub {

/**
 * A chunk of data inside a Hub
 */
template<typename Structure, typename BufferItem = void>
class Chunk {
public:

    /**
     * All envelopes on all devices that hold the same data (this chunk)
     */
    std::map<Device::index_t, Envelope<Structure, BufferItem>*> envelopes;

    Chunk(
        Envelope<Structure, BufferItem>& envelope,
        Device::index_t device_index
    ) {
        envelopes[device_index] = &envelope;
    }

    /**
     * Returns the optimal source envelope for data transfer to a given device
     */
    Envelope<Structure, BufferItem>& get_source_for_transfer(Device::index_t target) {
        assert(envelopes.count(target) == 0 && "Target is already present");

        // prefer copying "from host"
        if (envelopes.count(Device::HOST_INDEX) == 1)
            return *envelopes[Device::HOST_INDEX];

        // now we're copying "to host" and there's probably only one instance,
        // so pick the first that we come across
        assert(envelopes.size() > 0 && "The chunk has no envelopes to copy from");
        return *(envelopes.begin()->second);
    }
};

} // hub namespace
} // pipelines namespace
} // namespace noarr

#endif

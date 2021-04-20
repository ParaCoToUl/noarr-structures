#ifndef NOARR_PIPELINES_PORT_HPP
#define NOARR_PIPELINES_PORT_HPP

#include <cstddef>
#include "Envelope.hpp"
#include "UntypedPort.hpp"

namespace noarr {
namespace pipelines {

template<typename Structure, typename BufferItem = void>
class Port : public UntypedPort {
public:

    Port(Device::index_t device_index) : UntypedPort(device_index) { }

    /**
     * Returns a reference to the attached envelope
     */
    Envelope<Structure, BufferItem>& envelope() {
        return dynamic_cast<Envelope<Structure, BufferItem>&>(
            UntypedPort::envelope()
        );
    }
};

} // pipelines namespace
} // namespace noarr

#endif

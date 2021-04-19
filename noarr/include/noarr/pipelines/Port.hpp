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

    /**
     * Returns a reference to the attached envelope
     */
    Envelope<Structure, BufferItem>& get_envelope() {
        return dynamic_cast<Envelope<Structure, BufferItem>&>(
            this->get_untyped_envelope()
        );
    }
};

} // pipelines namespace
} // namespace noarr

#endif

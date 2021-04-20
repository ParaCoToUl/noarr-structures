#ifndef NOARR_PIPELINES_HUB_HPP
#define NOARR_PIPELINES_HUB_HPP

#include <cstddef>

#include "noarr/pipelines/UntypedPort.hpp"
#include "noarr/pipelines/Device.hpp"

#include "noarr/pipelines/Hub_BufferPool.hpp"

namespace noarr {
namespace pipelines {

/**
 * Envelope hub is responsible for:
 * - envelope lifecycle
 * - envelope routing
 * - moving data between devices
 * It is the glue between compute nodes.
 */
template<typename Structure, typename BufferItem = void>
class Hub {
private:
    std::size_t buffer_size;

public:
    // TODO: construction API to be figured out
    Hub(std::size_t buffer_count, std::size_t buffer_size) {
        this->buffer_size = buffer_size;
    }

    UntypedPort& write_port(Device dev) {
        // TODO: placeholder method
    }

    UntypedPort& read_port(Device dev) {
        // TODO: placeholder method
    }
};

} // pipelines namespace
} // namespace noarr

#endif

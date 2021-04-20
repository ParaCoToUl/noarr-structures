#ifndef NOARR_PIPELINES_DEVICE_HPP
#define NOARR_PIPELINES_DEVICE_HPP

namespace noarr {
namespace pipelines {

#include <cstddef>

/**
 * Represents a device that has memory
 * (host cpu or a gpu device)
 */
struct Device {
    using index_t = std::int8_t;

    index_t device_index = -1;

    // useful constants
    static const index_t HOST_INDEX = -1;
    static const index_t DEVICE_INDEX = 0;
    static const index_t DEVICE0_INDEX = 0;
    static const index_t DEVICE1_INDEX = 1;

    Device() {
        //
    }

    Device(index_t device_index) {
        this->device_index = device_index;
    }
};

} // pipelines namespace
} // namespace noarr

#endif

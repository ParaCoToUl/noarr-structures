#ifndef NOARR_PIPELINES_MEMORY_DEVICE_HPP
#define NOARR_PIPELINES_MEMORY_DEVICE_HPP

namespace noarr {
namespace pipelines {

/**
 * Represents a device that has memory
 * (host cpu or a gpu device)
 */
struct memory_device {
    unsigned char device_index = -1;

    memory_device() {
        //
    }

    memory_device(unsigned char device_index) {
        this->device_index = device_index;
    }
};

} // pipelines namespace
} // namespace noarr

#endif

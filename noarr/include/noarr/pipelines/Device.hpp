#ifndef NOARR_PIPELINES_DEVICE_HPP
#define NOARR_PIPELINES_DEVICE_HPP

namespace noarr {
namespace pipelines {

/**
 * Represents a device that has memory
 * (host cpu or a gpu device)
 */
struct Device {
    unsigned char device_index = -1;

    Device() {
        //
    }

    Device(unsigned char device_index) {
        this->device_index = device_index;
    }
};

} // pipelines namespace
} // namespace noarr

#endif

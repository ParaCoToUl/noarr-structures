#ifndef NOARR_PIPELINES_DEVICE_HPP
#define NOARR_PIPELINES_DEVICE_HPP

namespace noarr {
namespace pipelines {

/**
 * Represents a device that has memory
 * (host cpu or a gpu device)
 */
struct Device {
    using index_t = unsigned char;

    index_t device_index = -1;

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

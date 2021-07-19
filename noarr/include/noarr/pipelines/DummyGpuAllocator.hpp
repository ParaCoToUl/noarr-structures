#ifndef NOARR_PIPELINES_DUMMY_GPU_ALLOCATOR_HPP
#define NOARR_PIPELINES_DUMMY_GPU_ALLOCATOR_HPP

#include <vector>
#include <map>
#include <deque>
#include <iostream>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/HostAllocator.hpp"

namespace noarr {
namespace pipelines {

/**
 * Can allocate and free memory on the host
 */
class DummyGpuAllocator : public HostAllocator {
public:
    virtual Device::index_t device_index() const override {
        return Device::DUMMY_GPU_INDEX;
    };
};

} // pipelines namespace
} // namespace noarr

#endif

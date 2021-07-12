#ifndef NOARR_PIPELINES_HOST_ALLOCATOR_HPP
#define NOARR_PIPELINES_HOST_ALLOCATOR_HPP

#include <vector>
#include <map>
#include <deque>
#include <iostream>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/MemoryAllocator.hpp"

namespace noarr {
namespace pipelines {

/**
 * Can allocate and free memory on the host
 */
class HostAllocator : public MemoryAllocator {
public:
    virtual Device::index_t device_index() const override {
        return Device::HOST_INDEX;
    };

    virtual void* allocate(std::size_t bytes) const override {
        return malloc(bytes);
    };

    virtual void deallocate(void* buffer) const override {
        free(buffer);
    };
};

} // pipelines namespace
} // namespace noarr

#endif

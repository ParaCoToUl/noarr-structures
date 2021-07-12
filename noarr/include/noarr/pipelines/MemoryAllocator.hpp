#ifndef NOARR_PIPELINES_MEMORY_ALLOCATOR_HPP
#define NOARR_PIPELINES_MEMORY_ALLOCATOR_HPP

#include <vector>
#include <map>
#include <deque>
#include <iostream>

#include "noarr/pipelines/Device.hpp"

namespace noarr {
namespace pipelines {

/**
 * Can allocate and free memory on a single device
 */
class MemoryAllocator {
public:
    /**
     * Returns the device index, for which this allocator allocates
     */
    virtual Device::index_t device_index() const = 0;

    virtual void* allocate(std::size_t bytes) const = 0;

    virtual void deallocate(void* buffer) const = 0;
};

} // pipelines namespace
} // namespace noarr

#endif

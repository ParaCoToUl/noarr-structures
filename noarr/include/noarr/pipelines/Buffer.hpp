#ifndef NOARR_PIPELINES_BUFFER_HPP
#define NOARR_PIPELINES_BUFFER_HPP

#include <vector>
#include <map>
#include <deque>
#include <iostream>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/MemoryAllocator.hpp"

namespace noarr {
namespace pipelines {

/**
 * Buffer represents a continuous portion of memory on some device.
 * Destroying the object causes the underlying memory to be freed if it was
 * allocated during creation and not given as a constructor parameter.
 */
class Buffer {
public:
    Device::index_t device_index;
    std::size_t bytes;
    void* data_pointer;
    const MemoryAllocator* allocator;

private:
    Buffer(
        Device::index_t device_index,
        std::size_t bytes,
        void* existing_buffer,
        const MemoryAllocator* allocator
    ) :
        device_index(device_index),
        bytes(bytes),
        data_pointer(existing_buffer),
        allocator(allocator)
    { }

public:

    // disable copy-constructor and copy-assignment
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer& other) = delete;

    // empty buffer instance that holds no data
    Buffer() :
        device_index(Device::HOST_INDEX),
        bytes(0),
        data_pointer(nullptr),
        allocator(nullptr)
    { }

    // implement move-constructor
    Buffer(Buffer&& other) :
        device_index(other.device_index),
        bytes(other.bytes),
        data_pointer(other.data_pointer), 
        allocator(other.allocator)
    {
        other.bytes = 0;
        other.data_pointer = nullptr;
        other.allocator = nullptr;
    }

    // implement move-assignment
    Buffer& operator=(Buffer&& other) {
        if (this == &other)
            return *this;
        
        // free what we own currently
        if (allocator != nullptr && data_pointer != nullptr) {
            allocator->deallocate(data_pointer);
        }
        
        // this = other
        device_index = other.device_index;
        bytes = other.bytes;
        data_pointer = other.data_pointer;
        allocator = other.allocator;
        
        // other = null
        other.bytes = 0;
        other.data_pointer = nullptr;
        other.allocator = nullptr;

        return *this;
    }

    ~Buffer() {
        if (allocator != nullptr && data_pointer != nullptr) {
            allocator->deallocate(data_pointer);
        }
    }

    /**
     * Allocates a new buffer with a given allocator.
     * The memory WILL be released when this instance is destroyed.
     */
    static Buffer allocate_new(
        const MemoryAllocator& allocator,
        std::size_t bytes
    ) {
        return Buffer(
            allocator.device_index(),
            bytes,
            allocator.allocate(bytes),
            &allocator
        );
    }

    /**
     * Creates the buffer instance from an existing, allocated buffer.
     * The memory will NOT be released on this instance destruction.
     */
    static Buffer from_existing(
        Device::index_t device_index,
        void* existing_buffer,
        std::size_t bytes
    ) {
        return Buffer(
            device_index,
            bytes,
            existing_buffer,
            nullptr
        );
    }
};

} // pipelines namespace
} // namespace noarr

#endif

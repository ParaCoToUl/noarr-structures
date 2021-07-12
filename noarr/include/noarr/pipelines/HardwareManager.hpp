#ifndef NOARR_PIPELINES_HARDWARE_MANAGER_HPP
#define NOARR_PIPELINES_HARDWARE_MANAGER_HPP

#include <memory>
#include <cassert>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

#include "noarr/pipelines/Device.hpp"
#include "noarr/pipelines/Buffer.hpp"
#include "noarr/pipelines/MemoryAllocator.hpp"
#include "noarr/pipelines/HostAllocator.hpp"
#include "noarr/pipelines/MemoryTransferer.hpp"
#include "noarr/pipelines/HostTransferer.hpp"

namespace noarr {
namespace pipelines {

class HardwareManager; // forward declaration for the static variable

namespace {
    std::unique_ptr<HardwareManager> default_manager_instance;
}

/**
 * Hardware manager tracks and communicates with CPU and GPU devices.
 * It is primarily responsible for memory allocations and memory transfers.
 */
class HardwareManager {
public:
    /**
     * Returns the instance of the default hardware manager (singleton)
     * (this instance should be used in most cases, instead of creating a custom one)
     */
    static HardwareManager& default_manager() {
        if (default_manager_instance == nullptr) {
            default_manager_instance = std::make_unique<HardwareManager>();
        }

        return *default_manager_instance;
    }

    HardwareManager() {
        // register allocator and transferer for the host
        set_allocator_for(
            Device::HOST_INDEX,
            std::make_unique<HostAllocator>()
        );
        set_transferer_for(
            Device::HOST_INDEX, Device::HOST_INDEX,
            std::make_unique<HostTransferer>(false)
        );
    }

    /**
     * Registers a dummy GPU device that can be used for simulating memory
     * transfers on a system without any GPUs
     */
    void register_dummy_gpu() {
        set_allocator_for(
            Device::DUMMY_GPU_INDEX,
            std::make_unique<HostAllocator>()
        );
        set_transferer_for(
            Device::DUMMY_GPU_INDEX, Device::DUMMY_GPU_INDEX,
            std::make_unique<HostTransferer>(true)
        );
        set_transferer_for(
            Device::HOST_INDEX, Device::DUMMY_GPU_INDEX,
            std::make_unique<HostTransferer>(true)
        );
        set_transferer_for(
            Device::DUMMY_GPU_INDEX, Device::HOST_INDEX,
            std::make_unique<HostTransferer>(true)
        );
    }

    /**
     * Allocates a new buffer on the given device
     */
    Buffer allocate_buffer(Device::index_t device_index, std::size_t bytes) {
        MemoryAllocator& allocator = get_allocator_for(device_index);
        return Buffer::allocate_new(allocator, bytes);
    }

    /**
     * Transfers data between two buffers, typically on different devices.
     * The transfer is asynchronous, so a callback must be provided.
     */
    void transfer_data(
        Buffer from,
        Buffer to,
        std::size_t bytes,
        std::function<void()> callback
    ) {
        assert(bytes <= from.bytes && "Transfering too many bytes");
        assert(bytes <= to.bytes && "Transfering too many bytes");

        MemoryTransferer& transferer = get_transferer(
            from.device_index,
            to.device_index
        );

        transferer.transfer(
            from.data_pointer,
            to.data_pointer,
            bytes,
            callback
        );
    }

    /////////////////////
    // Lower-level API //
    /////////////////////

private:
    std::map<
        Device::index_t,
        std::unique_ptr<MemoryAllocator>
    > allocators;
    std::map<
        std::tuple<Device::index_t, Device::index_t>,
        std::unique_ptr<MemoryTransferer>
    > transferers;

public:

    /**
     * Returns allocator for a device
     */
    MemoryAllocator& get_allocator_for(Device::index_t device_index) {
        return *allocators[device_index];
    }

    /**
     * Sets allocator for a device
     */
    void set_allocator_for(
        Device::index_t device_index,
        std::unique_ptr<MemoryAllocator> allocator
    ) {
        allocators[device_index] = std::move(allocator);
    }

    /**
     * Returns transferer between two devices
     */
    MemoryTransferer& get_transferer(Device::index_t from, Device::index_t to) {
        return *transferers[std::make_tuple(from, to)];
    }

    /**
     * Sets a memory transferer for transfers from one given device to another
     * (only in that one direction)
     */
    void set_transferer_for(
        Device::index_t from,
        Device::index_t to,
        std::unique_ptr<MemoryTransferer> transferer
    ) {
        transferers[std::make_tuple(from, to)] = std::move(transferer);
    }
};

} // pipelines namespace
} // namespace noarr

#endif

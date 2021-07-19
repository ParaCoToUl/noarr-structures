#include <catch2/catch.hpp>

#include <string>
#include <memory>
#include <iostream>

#include <noarr/pipelines/Device.hpp>
#include <noarr/pipelines/Buffer.hpp>
#include <noarr/pipelines/HardwareManager.hpp>
#include <noarr/pipelines/DebuggingScheduler.hpp>
#include <noarr/pipelines/LambdaComputeNode.hpp>

using namespace noarr::pipelines;

TEST_CASE("Memory transfer", "[pipelines][unit][memory_transfer]") {
    auto& manager = HardwareManager::default_manager();
    manager.register_dummy_gpu();
    auto& transferer = manager.get_transferer(Device::HOST_INDEX, Device::DUMMY_GPU_INDEX);
    
    Buffer src_buffer = manager.allocate_buffer(Device::HOST_INDEX, 1024);
    Buffer dst_buffer = manager.allocate_buffer(Device::DUMMY_GPU_INDEX, 1024);

    int* src = (int*) src_buffer.data_pointer;
    int* dst = (int*) dst_buffer.data_pointer;

    src[0] = 1; // dummy data to be transfered
    src[1] = 2;
    src[2] = 3;

    bool transfer_completed = false;

    auto async_node = LambdaComputeNode(); {
        async_node.can_advance([&](){
            return !transfer_completed;
        });

        async_node.advance([&](){
            transferer.transfer(
                src_buffer.data_pointer,
                dst_buffer.data_pointer,
                1024,
                [&async_node](){
                    async_node.callback();
                }
            );
        });

        async_node.post_advance([&](){
            transfer_completed = true;
        });
    }

    auto scheduler = DebuggingScheduler();
    scheduler.add(async_node);
    scheduler.run();

    // check the data was transfered
    REQUIRE(dst[0] == 1);
    REQUIRE(dst[1] == 2);
    REQUIRE(dst[2] == 3);
}
